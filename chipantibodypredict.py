import argparse
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from numpy.ma.extras import unique
from scipy.ndimage import label
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from umap import UMAP

def add_umap_components(df, cols, n_components=3, perfix=""):

    umap = UMAP(n_components=n_components,
                n_neighbors=20,         # common default, can be tuned
                min_dist=0.01,           # common default, can be tuned
                metric='euclidean',random_state=4)     # can adjust based on your data
    umap.fit(df[cols])
    umap_cols = [f"{perfix}_umap{i}" for i in range(n_components)]
    df[umap_cols] = umap.transform(df[cols])
    return df, umap_cols

def add_tsn_components(df, cols, n_components=3,perfix=""):
    tsne = TSNE(n_components=n_components,method='exact',learning_rate=1,perplexity=30,metric='euclidean',random_state=42)
    tsne.fit(df[cols])
    tsne_cols = [f"{perfix}_tsne{i}" for i in range(n_components)]
    df[tsne_cols] = tsne.fit_transform(df[cols])
    return df, tsne_cols

def add_pca_components(df, cols, n_components=3,perfix=""):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(df[cols])
    pca_cols = [f"{perfix}_pca{i}" for i in range(n_components)]
    df[pca_cols] = pca.transform(df[cols])
    return df, pca_cols

def plot_pca(df,pca_cols,targetcol,title=""):
    fig,ax = plt.subplots(ncols=1,figsize=(10,5))
    labels = df[targetcol].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(labels)))
    for i,label in enumerate(labels):
        ax.scatter(df[df[targetcol]==label][pca_cols[0]],df[df[targetcol]==label][pca_cols[1]],color=colors[i],label=label)
    plt.legend()
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create table with statistics of bigwig files")
    parser.add_argument("-incsv", type=str, help="statistic file",default="house_keeping_stat_3c.csv")
    parser.add_argument("--idcol", type=str, help="", default="filename")
    parser.add_argument("--targetcol", type=str, help="", default="antibody_id")
    parser.add_argument("-outcsv", type=str, help="Path to output CSV", default="stats_predict.csv")
    parser.add_argument("--verbose", type=int, help="Verbose level", default=2)

    args = parser.parse_args()
    data = pd.read_csv(args.incsv)
    idcol = args.idcol
    targetcol = args.targetcol
    outcsv = args.outcsv
    verbose = args.verbose

    #check if target column is in data
    if targetcol not in data.columns:
        print(f"Error: Column '{targetcol}' does not exist.")
        exit(1)
    #check if id column is in data
    if idcol not in data.columns:
        print(f"Error: Column '{idcol}' does not exist.")
        exit(1)
    #get list of all other cols in data

    othercols = [col for col in data.columns if col not in [idcol,targetcol]]
    #remove not numeric cols from othercols
    othercols = [col for col in othercols if pd.api.types.is_numeric_dtype(data[col])]

    #Add additional cols by reduction of dimensionality
    additional_cols = []
    list_of_cols_prefix = ['ACF','MF','q']
    list_of_methods_of_reduction = ['pca','tsne','umap']
    number_of_additional_components = 4
    for prefix in list_of_cols_prefix:
        for method in list_of_methods_of_reduction:
            cols = [col for col in othercols if col.startswith(prefix)]
            if len(cols) > 0:
                if method == 'pca':
                    data, pca_cols = add_pca_components(data, cols, n_components=number_of_additional_components,perfix=prefix)
                    additional_cols += pca_cols
                if method == 'tsne':
                    data, tsne_cols = add_tsn_components(data, cols, n_components=number_of_additional_components,perfix=prefix)
                    additional_cols += tsne_cols
                if method == 'umap':
                    data, umap_cols = add_umap_components(data, cols, n_components=number_of_additional_components,perfix=prefix)
                    additional_cols += umap_cols

    othercols += additional_cols

    #get list of rows with value in target column
    trainrows = data[data[targetcol].notnull()]
    #get list of rows with no value in target column
    testrows = data[data[targetcol].isnull()]


    # Separate features (X) and target (y) for training
    X = trainrows[othercols]
    y = trainrows[targetcol]

    # Encode the target labels (since they are strings)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # (Optional) Split into train/validation sets to check performance before final training
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=15)
    # Initialize and train the RandomForest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=50)
    clf.fit(X_train, y_train)
    # (Optional) Evaluate on validation set
    val_predictions = clf.predict(X_val)
    val_accuracy = (val_predictions == y_val).mean()
    #calculate accuracy by class
    print(pd.DataFrame({"true":label_encoder.inverse_transform(y_val),"pred":label_encoder.inverse_transform(val_predictions)}))
    print(pd.DataFrame({"true":label_encoder.inverse_transform(y_val),"pred":label_encoder.inverse_transform(val_predictions)}).groupby(["true","pred"]).size())

    if verbose:
        print(f"Validation Accuracy: {val_accuracy:.2f}")

    # Show top of most important features
    if verbose:
        feature_importances = pd.Series(clf.feature_importances_, index=othercols)
        print("Top 10 most important features:")
        print(feature_importances.nlargest(10))

    # top best predicted classes
    if verbose:
        print("Top 5 best predicted classes:")
        print(pd.Series(clf.predict_proba(X_val).max(axis=1)).nlargest(5))  # top best predicted classes
        #

    if len(testrows) == 0:
        exit(0)
    # Once satisfied, retrain on all available training data (optional step)
    # This can slightly improve performance since we are using all data now.
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_encoded)



    # Predict on testrows where the target is unknown
    X_test = testrows[othercols]
    test_pred_encoded = clf.predict(X_test)

    # Convert encoded predictions back to original string labels
    test_pred_labels = label_encoder.inverse_transform(test_pred_encoded)

    # Store predictions in testrows
    testrows[targetcol] = test_pred_labels

    # Optionally combine trainrows and testrows or just output test predictions
    # If you only need the test predictions:
    if outcsv:
        testrows[[idcol, targetcol]].to_csv(outcsv, index=False)
        if verbose:
            print(f"Predictions saved to {outcsv}")
