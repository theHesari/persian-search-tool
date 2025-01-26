import pandas as pd
from tqdm import tqdm
from chromadb import Client
from chromadb.config import Settings
from dadmatools.normalizer import Normalizer

normalizer = Normalizer()

def initialize_chromadb(collection_name):
    """
    Initialize the ChromaDB client and collection.
    
    Args:
        collection_name (str): Name of the ChromaDB collection.
    
    Returns:
        collection: The ChromaDB collection object.
    """
    client = Client(Settings(
        persist_directory="chromadb_storage",  # Directory to store the database
        chroma_db_impl="duckdb+parquet"  # Use DuckDB (SQLite-like) with Parquet for storage
    ))
    return client.get_or_create_collection(name=collection_name)

def create_batches(df, batch_size):
    """
    Split the DataFrame into batches.
    
    Args:
        df (pd.DataFrame): The DataFrame to split.
        batch_size (int): The size of each batch.
    
    Yields:
        pd.DataFrame: A batch of the DataFrame.
    """
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

def insert_batch_into_chromadb(collection, batch):
    """
    Insert a single batch into the ChromaDB collection.
    
    Args:
        collection: The ChromaDB collection object.
        batch (pd.DataFrame): A batch of data to insert.
    """
    documents = batch["title_fa"].tolist()
    ids = batch["id"].tolist()
    metadatas = batch[["Category1", "sub_category"]].to_dict(orient="records")
    
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    # print(f"Inserted batch with IDs: {ids}")

def batch_insert_to_chromadb(collection_name, data, batch_size):
    """
    Perform batch insertion of data into a ChromaDB collection.
    
    Args:
        collection_name (str): The name of the ChromaDB collection.
        data (pd.DataFrame): The DataFrame containing the data.
        batch_size (int): The size of each batch for insertion.
    """
    collection = initialize_chromadb(collection_name)
    total_batches = (len(data) + batch_size - 1) // batch_size  # Calculate the number of batches
    
    with tqdm(total=total_batches, desc="Inserting Batches") as pbar:
        for batch in create_batches(data, batch_size):
            insert_batch_into_chromadb(collection, batch)
            pbar.update(1)

def clean(dataframe):
    """
    Clean the DataFrame by removing duplicates and resetting the index.
    Unify the characters in the "title_fa" and "Category1" column.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to clean.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    print("Preparing data ...")
    dataframe = dataframe.drop_duplicates(subset="id").reset_index(drop=True)
    dataframe["title_fa"] = dataframe["title_fa"].apply(normalizer.normalize)
    dataframe["Category1"] = dataframe["Category1"].apply(normalizer.normalize)
    print("Data is cleaned and normalized.")
    return dataframe
# Example usage
if __name__ == "__main__":

    # Dynamic inputs
    data_path = input("Enter your data directory: ")
    collection_name = input("Enter the ChromaDB collection name: ")
    batch_size = int(input("Enter the batch size: "))

    # Define your DataFrame
    df = pd.read_csv(data_path, dtype=str)
    data = clean(df)

    # Perform batch insertion
    batch_insert_to_chromadb(collection_name, data, batch_size)
