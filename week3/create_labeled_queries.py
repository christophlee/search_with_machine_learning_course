import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.

# convert to lowercase
queries_df['query_normalized'] = queries_df['query'].str.lower()

# replace non-alphanumeric with space
queries_df['query_normalized'] = queries_df['query_normalized'].str.replace(r'[^0-9a-z]',' ',regex=True)

# remove excess whitespace and stem each token
queries_df['query_normalized'] = queries_df.apply(
    lambda x: ' '.join([stemmer.stem(token.strip()) for token in x['query_normalized'].split()]),
    axis=1
)

# count queries per category
# count_per_category = queries_df.category.value_counts()

# check for correctness
# assert(count_per_category.loc['abcat0701001'] == 13830)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
parents_df_reindexed = parents_df.set_index('category')

queries_df['pruned_category'] = queries_df['category']
queries_df['pruned_count'] = 0

# iterate until no more categories to prune
while True:

    # count number of queries per category
    counts = queries_df.pruned_category.value_counts()

    # select all categories with fewer than min_queries (but don't include root category)
    categories_to_prune = list(counts[(counts < min_queries) & (counts.index != root_category_id)].index)

    if categories_to_prune == []:
        break

    # select all queries belonging to these categories
    queries_to_roll_up = queries_df[queries_df.pruned_category.isin(categories_to_prune)]

    print (f'pruning {len(categories_to_prune)} categories with {len(queries_to_roll_up)} associated queries...')

    # update category for these queries to parent category
    queries_df.loc[queries_to_roll_up.index, 'pruned_category'] = parents_df_reindexed.loc[queries_to_roll_up.pruned_category].parent.values
    queries_df.loc[queries_to_roll_up.index, 'pruned_count'] += 1

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['pruned_category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['pruned_category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
