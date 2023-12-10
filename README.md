# Study-of-Dimensionality-Reduction-Techniques-and-Interpretation-of-their-Coefficients

Code used for my Masters Thesis "Study of Dimensionality Reduction Techniques and Interpretation of their Coefficients, and Influence on Learned Models", which can be accessed [here](https://oa.upm.es/75893/). It obtained the maximum grade (10/10).

## What was done

First, the dimensionality of the data was reduced using state-of-the-art dimensionality reduction techniques such as SLMVP. These reduction techniques were combined with different machine learning classifiers to fine-tune their parameters. The objective was to identify the optimal configuration that achieves the highest accuracy with the given data. The accuracy obtained with only the first k components is measured for different values of k.

Second, the performance of the techniques in capturing and preserving the structure of the original dataset is analyzed by plotting their projections in 2 and 3-dimensional plots. We look into whether the data points are evenly distributed or not, this shows how effectively the technique has managed to capture the overall variance of the dataset, and whether the graph exhibits a clear separation of the different classes. This, paired with the accuracy obtained in the previous classification task, tells us about the goodness of the technique.

Finally, the correlations between the original data and each one of the components obtained through dimensionality reduction are leveraged to extract meaningful qualitative information. This is based on the fact that the components are the directions of maximum variability of the data and it is fair to assume that the variables that have a high absolute correlation with a component are given a high significance by the dimensionality reduction technique. A recommendation is then given as to which features should be selected for a posterior machine learning task, based on their absolute correlation with the components.

In addition, the correlations are also leveraged to compare the similarity and dissimilarity of components realized by applying different techniques. This is done by calculating the spearman correlation coefficient of the absolute correlation between two components, obtaining a similarity score.

## Theoretical principles

This work draws inspiration mainly from the following papers:

> Esteban García-Cuesta, José Antonio Iglesias.
> ["User modeling: Through statistical analysis and subspace learning."](https://doi.org/10.1016/j.eswa.2011.11.015)

> Jolliffe, I. T.
>["Discarding Variables in a Principal Component Analysis."](https://doi.org/10.2307/2346488)

<!-- ## License
GNU GENERAL PUBLIC LICENSE Version 3 -->

## File structure

- __`main.ipynb`__: IPython notebook showing the pipeline and the results.
- __`models.py`__: Contains packaged code to train and test the models, as well as generate graphs and tables.
- `slmvp.py`: Dimensionality Reduction Technique SLMVP.
- `datasets.py`: code to load and prepare data for pipeline.
- `requirements.txt`
- `TFM-Doc/`: LaTex documents used to create the thesis document.
