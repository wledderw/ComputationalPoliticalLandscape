import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def request(address):
    """
    Function for making http requests, and immediately parsing boldface.

    Parameters:
    address: web address to scrape, as string.

    Return: 
    the web content as a string.
    """
    x = BeautifulSoup(requests.get(address, timeout=120).text, 'html.parser')

    # Keep the boldface tags in the plaintext
    soup = str(x).replace("<strong>", "#####").replace("</strong>", "$$$$$")

    # Process using BeautifulSoup to get rid of all HTML tags
    text = BeautifulSoup(soup, 'html.parser').get_text()

    # Start parsing from "De ###voorzitter$$$:Ik start de vergadering.\n",
    # Stop parsing from "Sluiting xx.xx uur."
    start_idx = re.search(r"De #####voorzitter\$\$\$\$\$:.+\.\n", text).end()
    end_idx = re.search(r"Sluiting \d+\.\d+ uur.", text).start()

    return text[start_idx:end_idx]


def preprocess(text):
    """
    Function for preprocessing the the web pages.
    Highlights any persons.

    Parameters:
    text: the web page, with parsed boldface.

    Return: 
    preprocessed web content as a string.
    """
    # Add @@@ at the start of a speech.
    text = re.sub(r'De #####voorzitter\$\$\$\$\$:', r'@@@De ###voorzitter$$$:', text)
    text = re.sub(r'Minister #####(.+)\$\$\$\$\$:', r'@@@Minister ###\1$$$:', text)
    text = re.sub(r'Staatssecretaris #####(.+)\$\$\$\$\$:', r'@@@Staatssecretaris ###\1$$$:', text)
    text = re.sub(r'Mevrouw #####(.+)\$\$\$\$\$ \((.+)\):', r'@@@Mevrouw ###\1$$$ (\2):', text)
    text = re.sub(r'De heer #####(.+)\$\$\$\$\$ \((.+)\):', r'@@@De heer ###\1$$$ (\2):', text)
    text = re.sub(r'Kamerlid #####(.+)\$\$\$\$\$ \((.+)\):', r'@@@Kamerlid ###\1$$$ (\2):', text)

    # Remove any other bold that is not related to a person:
    text = re.sub(r'#####(.*)\$\$\$\$\$', r'\1', text)

    # Change enters into space
    text = re.sub(r'(\n)+', r' ', text)
    text = re.sub(r' +', r' ', text)

    return text


def parse_into_df(df, preprocessed_list, ministers, parties, all_lengths):
    """
    Function for parsing speeches into a Pandas dataframe.

    Parameters: 
    df: the empty dataframe, consisting of a Text, Party and Speaker column.
    preprocessed_list: a list, where each entry in the list is a new speech (with speaker and party)
    ministers: a dictionary of minister names and their affiliated party
    parties: a list of the parties
    all_lengths: empty list to collect lengths of speeches in

    Return:
    the dataframe filled with speeches and their respective speakers and parties
    the lengths of the speeches
    """
    for speech in preprocessed_list:
        if speech is None:
            continue
        if "De ###voorzitter$$$:" in speech:
            continue  # Skip the chairman
        text_index = re.search(r":", speech).end()
        text = speech[text_index:]
        if len(re.findall(r"[a-zA-Z0-9]+(?:['’-][a-zA-Z0-9]+)*", text)) < 15:
            continue  # Skip speeches shorter than 15 words

        all_lengths.append(len(re.findall(r"[a-zA-Z0-9]+(?:['’-][a-zA-Z0-9]+)*", text)))

        match = re.search(r"(Minister|Staatssecretaris) ###(.+)\$\$\$", speech)

        # In case of Minister or Staatssecretaris:
        if match:
            name = match.group(2)
            person = match.group(1) + " " + name
            party = ministers[name] # Find party using dictionary:

        # In case of De heer, Mevrouw or Kamerlid:
        else:
            match = re.search(r"(De heer|Mevrouw|Kamerlid) ###(.+)\$\$\$ \((.+)\):", speech)
            person = match.group(1) + " " + match.group(2)
            party = match.group(3)
            if party not in parties:
                continue
        df = pd.concat([df, pd.DataFrame({'Text': [text], 'Party':[party], 'Speaker': [person]})], ignore_index=True)

    return df, all_lengths


if __name__ == "__main__": 
    # Make for-loop over all 2024-2025 1-28 and 2023-2024 July 2nd - the end
    addresses2324 = ["https://www.tweedekamer.nl/kamerstukken/plenaire_verslagen/detail/2023-2024/"+str(i) for i in range(91, 99)]
    addresses2425 = ["https://www.tweedekamer.nl/kamerstukken/plenaire_verslagen/detail/2024-2025/"+str(i) for i in range(1, 90)]
    addresses = addresses2324 + addresses2425

    # Initialize dataframe, ministers and parties
    df = pd.DataFrame({'Text':[], 'Party':[], 'Speaker':[]})
    ministers = {'Schoof': 'Schoof', 'Agema': 'PVV', 'Hermans': 'VVD',
                 'Van Hijum': 'NSC', 'Keijzer': 'BBB', 'Uitermark': 'NSC',
                 'Veldkamp': 'NSC', 'Van Weel': 'VVD', 'Heinen': 'VVD',
                 'Beljaarts': 'PVV', 'Brekelmans': 'VVD', 'Bruins': 'NSC',
                 'Madlener': 'PVV', 'Wiersma': 'BBB', 'Faber': 'PVV',
                 'Klever': 'PVV', 'Szabó': 'PVV', 'Van Marum': 'BBB',
                 'Idsinga': 'NSC', 'Van Oostenbruggen': 'NSC', 'Achahbar': 'NSC',
                 'Palmen': 'NSC', 'Tuinman': 'BBB', 'Maeijer': 'PVV',
                 'Karremans': 'VVD', 'Nobel': 'VVD', 'Coenradie': 'PVV',
                 'Struycken': 'NSC', 'Paul': 'VVD', 'Jansen': 'PVV', 'Rummenie': 'BBB'
                }
    parties = ('PVV', 'GroenLinks-PvdA', 'JA21', 'BBB', 'D66', 'Volt', 'ChristenUnie', 'CDA', 'SP', 'DENK', 'PvdD', 'FVD', 'SGP', 'VVD', 'NSC')
    all_lengths = []

    # Start scraping
    for address in addresses:
        preprocessed_text = preprocess(request(address))
        preprocessed_list = preprocessed_text.split("@@@")[1:]
        df, all_lengths = parse_into_df(df, preprocessed_list, ministers, parties, all_lengths)

    # Compute and plot lengths of speeches:
    mean_length = np.mean(all_lengths)
    print("Mean length of speeches:", mean_length)

    plt.figure(figsize=(10,6))
    plt.hist(all_lengths, bins=range(0,3125,25))
    plt.axvline(mean_length, c='r', label='Mean length')
    plt.legend()
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.title("Speech lengths. Each bin has width 25.")
    plt.savefig(os.path.join(os.getcwd(), "plots/samples_lengths.png"))

    # Compute and plot amount of speeches per party:
    party_counts = df['Party'].value_counts()
    print(party_counts)

    party_counts = pd.concat([party_counts.drop('Schoof'), party_counts.loc[['Schoof']]])
    parties = ['PVV', 'VVD', 'NSC', 'GL-PvdA', 'D66', 'BBB', 'SP', 'CU', 'CDA', 'DENK', 'SGP', 'PvdD', 'Volt', 'FVD', 'JA21', 'Schoof']
    counts = party_counts.values
    
    plt.figure(figsize=(26,18))
    plt.grid(axis='y')
    plt.bar(parties, counts, color='#be311e', edgecolor='black')
    plt.xlabel('Party', fontsize=50)
    plt.xticks(rotation=90, ha='center', fontsize=50)
    plt.ylabel('Amount of speech samples', fontsize=50)
    plt.yticks(fontsize=50)
    plt.title('Histogram of speech samples per party', fontsize=60)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "plots/samples_parties_counts.png"))

    # Save full dataframe:
    df.to_csv(path_or_buf=os.path.join(os.getcwd(), "data/data.csv"), index=False)
    