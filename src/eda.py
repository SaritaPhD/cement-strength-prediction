import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(df: pd.DataFrame):
    """
    Perform Exploratory Data Analysis (EDA) on the given DataFrame.

    This function visualizes the distributions of each column in the DataFrame using
    histograms and detects outliers using boxplots.

    - The first set of plots is histograms, showing the distribution of values for
      each column.
    - The second set of plots is boxplots, highlighting potential outliers.

    Args:
        df (pd.DataFrame): The input DataFrame on which to perform EDA.

    Returns:
        None: The function displays plots but does not return any value.
    """
    
    # Visualize distributions using histograms
    plt.figure(figsize=(15, 15))  
    plotnumber = 1
    for i in df.columns:
        ax = plt.subplot(4, 3, plotnumber)  
        sns.histplot(df[i])  
        plt.xlabel(i, fontsize=10)  
        plotnumber += 1
    plt.tight_layout() 
    plt.show()  

    # Check for outliers using boxplots
    plt.figure(figsize=(15, 15))  
    plotnumber = 1
    for i in df.columns:
        ax = plt.subplot(4, 3, plotnumber)  
        sns.boxplot(df[i])  
        plt.xlabel(i, fontsize=10)  
        plotnumber += 1
    plt.tight_layout()  
    plt.show()  
