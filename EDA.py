# %% [markdown]
# # **EDA of new extraction of PubChemQC**
# 
# **Responsible** for the extraction: Vinicius Ávila
# 
# **Version**: v2
# 
# **Date of extraction**: `April 2023`
# 
# **Objective**: Make an exploratory data analysis (EDA) of the data
# 
# **Details**: available at the README file

# %% [markdown]
# # Importing libraries and configuration

# %%
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from rdkit import Chem
import matplotlib.pyplot as plt

# %%
tqdm.pandas()

# %%
pd.set_option('display.max_columns', None)

# %%
sns.set_theme(style="darkgrid", palette="husl", rc={"figure.figsize":(10, 5)})

# %% [markdown]
# # Loading data

# %%
df_smiles_1 = pd.read_parquet("smiles_list.parquet")

# %%
df_smiles_2 = pd.read_parquet("pubchem_smiles.parquet")

# %%
df_properties = pd.read_parquet("qm_properties.parquet")

# %%
df_transitions = pd.read_parquet("transitions_energies.parquet")

# %% [markdown]
# # EDA

# %% [markdown]
# ## Df smiles

# %%
df_smiles_1 = df_smiles_1.reset_index()
df_smiles_1

# %%
df_smiles_2

# %%
df_smiles_1.info()

# %%
df_smiles_2.isnull().sum()

# %% [markdown]
# Dataset 1 information (`smiles_list`)

# %%
df_smiles_1['cid'].duplicated().sum()

# %%
df_smiles_1['smiles'].duplicated().sum()

# %%
df_smiles_1['smiles source'].value_counts()

# %% [markdown]
# Dataset 2 information (`pubchem_smiles`)

# %%
df_smiles_2['cid'].duplicated().sum()

# %%
df_smiles_2['canonical smiles'].duplicated().sum()

# %%
df_smiles_2['isomeric smiles'].duplicated().sum()

# %%
df_smiles_2['isomeric smiles'].value_counts()

# %% [markdown]
# Let us follow only with the second dataset and use the isomeric smiles

# %%
df_smiles_2 = df_smiles_2[['cid', 'isomeric smiles']]

# %%
df_smiles_2 = df_smiles_2.rename(columns={'isomeric smiles':'isomeric_smiles'})

# %% [markdown]
# ## Df properties

# %%
df_properties = df_properties.reset_index()

# %%
new_col_names = dict(zip(df_properties.columns,["_".join(col.split()) for col in df_properties.columns]))
new_col_names

# %%
df_properties = df_properties.rename(columns=new_col_names)

# %%
df_properties.head()

# %%
df_properties.info()

# %%
df_properties.isnull().sum()

# %%
# PLOTS - Properties
fig, axs = plt.subplots(4,2, figsize=(9,7))
fig.tight_layout()

df_properties['charge'].plot.hist( ax=axs[0,0], title='charge')
df_properties['total_dipole_moment'].plot.hist( ax=axs[0,1], title='total_dipole_moment')
df_properties['multiplicity'].plot.hist( ax=axs[1,0], title='multiplicity')
df_properties['homo'].plot.hist( ax=axs[1,1], title='homo')
df_properties['lumo'].plot.hist( ax=axs[2,0], title='lumo')
df_properties['gap'].plot.hist( ax=axs[2,1], title='gap')
df_properties['total_energy'].plot.hist( ax=axs[3,0], title='total_energy')

# %% [markdown]
# ## Df transitions

# %%
df_transitions.head()

# %%
df_transitions.describe()

# %%
df_transitions[[x for x in df_transitions.columns if 'ET' in x]].describe().T['mean'].plot.bar(title='Mean Transition Energies')

# %%
df_transitions[[x for x in df_transitions.columns if 'OS' in x]].describe().T['mean'].plot.bar(title='Mean Oscillation Frequencies')

# %%
df_transitions = df_transitions.reset_index()

# %%
df_transitions.shape

# %%
df_transitions.info()

# %%
df_transitions.isnull().sum()

# %%
# PLOTS - Excitation energies
fig, axs = plt.subplots(5,2, figsize=(10,8))
fig.suptitle('Transition energies', weight ='bold')
fig.tight_layout()

df_transitions['TD_ET_00'].plot.hist(title='TD_ET_00', ax=axs[0,0])
df_transitions['TD_ET_01'].plot.hist(title='TD_ET_01', ax=axs[0,1])
df_transitions['TD_ET_02'].plot.hist(title='TD_ET_02', ax=axs[1,0])
df_transitions['TD_ET_03'].plot.hist(title='TD_ET_03', ax=axs[1,1])
df_transitions['TD_ET_04'].plot.hist(title='TD_ET_04', ax=axs[2,0])
df_transitions['TD_ET_05'].plot.hist(title='TD_ET_05', ax=axs[2,1])
df_transitions['TD_ET_06'].plot.hist(title='TD_ET_06', ax=axs[3,0])
df_transitions['TD_ET_07'].plot.hist(title='TD_ET_07', ax=axs[3,1])
df_transitions['TD_ET_08'].plot.hist(title='TD_ET_08', ax=axs[4,0])
df_transitions['TD_ET_09'].plot.hist(title='TD_ET_09', ax=axs[4,1])

# %%
# PLOTS - Oscillator forces
fig, axs = plt.subplots(5,2, figsize=(10,8))
fig.suptitle('Oscilation energies', weight ='bold')
fig.tight_layout()

df_transitions['TD_OS_00'].plot.hist(title='TD_OS_00', ax=axs[0,0])
df_transitions['TD_OS_01'].plot.hist(title='TD_OS_01', ax=axs[0,1])
df_transitions['TD_OS_02'].plot.hist(title='TD_OS_02', ax=axs[1,0])
df_transitions['TD_OS_03'].plot.hist(title='TD_OS_03', ax=axs[1,1])
df_transitions['TD_OS_04'].plot.hist(title='TD_OS_04', ax=axs[2,0])
df_transitions['TD_OS_05'].plot.hist(title='TD_OS_05', ax=axs[2,1])
df_transitions['TD_OS_06'].plot.hist(title='TD_OS_06', ax=axs[3,0])
df_transitions['TD_OS_07'].plot.hist(title='TD_OS_07', ax=axs[3,1])
df_transitions['TD_OS_08'].plot.hist(title='TD_OS_08', ax=axs[4,0])
df_transitions['TD_OS_09'].plot.hist(title='TD_OS_09', ax=axs[4,1])

# %% [markdown]
# # Joining datasets 
# 
# per cid index

# %%
df_smiles_2 = df_smiles_2.set_index('cid')
df_properties = df_properties.set_index('cid')
df_transitions = df_transitions.set_index('cid')

# %%
df_joined = df_smiles_2.join((df_properties.join(df_transitions, on='cid', how='left')), on='cid', how='left')

# %%
df_joined.head()

# %%
df_joined.shape

# %%
df_joined.isnull().sum()

# %%
df_joined.plot.scatter(x='gap', y='TD_ET_00')

# %%
df_joined_numeric = df_joined.drop(columns=['isomeric_smiles'])

# %%
sns.heatmap(df_joined_numeric, annot=True)

# %% [markdown]
# Dropping null data

# %%
df_joined = df_joined.dropna()

# %% [markdown]
# # SMILES check

# %%
df_joined[df_joined['isomeric_smiles'].duplicated()==True].head()

# %%
from rdkit.Chem import Draw
opts = Draw.DrawingOptions()
Draw.SetComicMode(opts)

# %%
mol1 = Chem.MolFromSmiles('C[N+]1(CC(CC1C(=O)[O-])O)C', sanitize=False)
display(Draw.MolToImage(mol1, size=(400, 200), fitImage=True))

# %%
df_joined[df_joined['isomeric_smiles']=='C[N+]1(CC(CC1C(=O)[O-])O)C']

# %% [markdown]
# Although they are isomers, their properties are not the same, so we will not drop them from the dataset

# %% [markdown]
# # Saving

# %%
df_joined.to_parquet('joined_data.parquet')


