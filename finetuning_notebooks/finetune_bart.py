import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class PaperSummaryDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BartTokenizer, text_max_token_len: int = 512,
                 summary_max_token_len: int = 150):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row["processed_text"]
        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt")

        summary = data_row["summary"]
        summary_encoding = self.tokenizer(
            summary,
            max_length=self.summary_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt")

        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            text=text,
            summary=summary,
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten())


class PaperSummaryDataModule(pl.LightningDataModule):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer: BartTokenizer, batch_size: int = 8,
                 text_max_token_len: int = 512, summary_max_token_len: int = 128):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage=None):
        self.train_dataset = PaperSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        self.test_dataset = PaperSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True)


class PaperSummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", return_dict=True)

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask)
        return output.loss, output.logits

    def step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self.forward(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     decoder_attention_mask=labels_attention_mask,
                                     labels=labels)
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, outputs = self.step(batch, batch_idx)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs = self.step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.00003, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=10000)
        return [optimizer], [scheduler]


def summarize(text, model, tokenizer):
    model.model.eval()  # Put the model in evaluation mode
    text_encoding = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    text_encoding = {k: v.to(model.device) for k, v in text_encoding.items()}

    generated_ids = model.model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=150,  # Adjust as needed
        num_beams=4,  # Tune these parameters as needed
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for gen_id in
        generated_ids
    ]
    return "".join(preds)


def main():
    pl.seed_everything(42)

    df = pd.read_csv('final.csv')
    df = df[['text', 'summary', 'processed_text']]
    df = df.dropna()

    train_df, test_df = train_test_split(df, test_size=0.1)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = PaperSummaryModel()

    data_module = PaperSummaryDataModule(train_df, test_df, tokenizer, batch_size=2)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping],
        accumulate_grad_batches=4
    )

    trainer.fit(model, data_module)

    trained_model = PaperSummaryModel.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )

    sample_row = test_df.iloc[5]
    text = sample_row["processed_text"]
    ref_summary = sample_row["summary"]

    model_summary = summarize(text, trained_model, tokenizer)

    print('Original text: \n', text)
    print('\nPredicted summary: \n', model_summary)
    print('\nOriginal summary: \n', ref_summary)


if __name__ == '__main__':
    main()
