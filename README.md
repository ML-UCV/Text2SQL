# Text2SQL

The scope of the given repository is to offer a tool that translates natural language into SQL queries. The program takes questions as input and returns the corresponding query.
The base of the project is RAT-SQL (https://github.com/Microsoft/rat-sql) further enhanced by Roberta Base Squad2 (https://huggingface.co/deepset/roberta-base-squad2?context=Which+students+play+DFFS%3F&question=Game+is+what%3F).

A few attempts were made using GPT-NEO (https://huggingface.co/docs/transformers/model_doc/gpt_neo) but due to the low score (45.37%), Roberta was prefered due to the better performance (74.15%).
