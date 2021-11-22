""" Script to grab text to annotate for product ids NER"""

# imports
import pandas as pd

def customer_messages(topic="Shipping", project="questrom"):
    """Generate a csv of messages to use for custom NER tasks

    Args:
        topic (str, optional): The topic to use in the WHERE clause. Defaults to "Shipping".
        project (str, optional): Billing project for Google Cloud. Defaults to "questrom".
    """
    SQL = f"SELECT text FROM `questrom.datasets.topics` WHERE topic = '{topic}'"
    df = pd.read_gbq(SQL, project)
    # write to csv
    df = df.sample(100, random_state=820)
    df.text.to_csv("messages.csv", index=False)


if __name__ == "__main__":
    # TODO: cli args via click or some other tool
    customer_messages()