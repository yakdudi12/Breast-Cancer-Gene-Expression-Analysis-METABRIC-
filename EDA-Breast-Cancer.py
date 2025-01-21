import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sqlite3
import math

class EDA:
    def __init__(self,path,type,sql_query=None):
        self.df = None
        try:
            if type == 'csv':
                self.df = pd.read_csv(path)
                print(f"{'-'*5}The CSV file was successfully loaded{'-'*5}")
            elif type == 'xls' or type == 'xlsx' or type == 'xlsm' or type == 'excel':
                self.df = pd.read_excel(path)
                print(f"{'-'*5}The Excel file was successfully loaded.{'-'*5}")
            elif type == 'sqlite' and sql_query or type == "sql" and sql_query:
                connect = sqlite3.connect(path)
                self.df = pd.read_sql_query(sql_query,connect)
                print(f"{'-'*5}The SQL file was successfully loaded{'-'*5}")
            elif type == 'dataframe' or type == 'pd':
                self.df = path
            else:
                raise ValueError("The file type is not acceptable or the SQL query is missing")
        except FileNotFoundError:
            print(f"ERROR: File not found: {path}")
        except sqlite3.Error as e:
            print(f"ERROR trying to execute the SQLite query: {e}")
        except Exception as e:
            print(f"Unexpected Error trying to load the dataset: {e}")

    '''Data treatment Methods'''
    def show_columns_types(self):
        print(f"\n{'-'*5}The dataset contains this information:{'-'*5}\n")
        print(self.df.info())

    def missing_values(self):
        if self.df.isnull().sum().sum() <= 0:
            print(f"\n{'-'*5}There are no null values in the dataset{'-'*5}")
        elif self.df.isnull().sum().sum() > 0:
            print("\nPercentage of null values in each column:")
            print((self.df.isnull().sum() / self.df.shape[0]) * 100)
            # Heatmap:
            plt.figure(figsize=(10, 5))
            corr = self.df.isnull()
            sns.heatmap(corr, cbar=True, cmap='cividis')
            plt.title("Null values heatmap in the dataset")
            plt.show()

    def detect_duplicates(self):
        duplicated_rows = self.df.duplicated()
        if any(duplicated_rows) is False:
            print(f"\n{'-'*5}There are no duplicated rows in the dataset.{'-'*5}")
        else:
            print(f"\n{'-'*5}There are duplicate rows in the dataset{'-'*5}\n")
            print(self.df.duplicated().value_counts())
            self.df.drop_duplicates(inplace=True)
            print(f"{'-'*5}The duplicate rows were successfully deleted{'-'*5}")

    def show_first_last_row(self):
        print(f"\nFirst rows:\n{self.df.head()}\n"
              f"\nLast rows:\n{self.df.tail()}\n")


    '''EDA-Methods'''
    def calculate_basic_values(self):
        try:
            print(f"Summary of Numerical Variables:\n {self.df.describe()}")
            print(f"\nSummary of Categorical Variables:\n{self.df.describe(include=['object', 'category'])}")
            print(f"\nMost Common Elements in Categorical Variables:\n{self.df.select_dtypes(include=['object', 'category']).mode().iloc[0]}")
        except ValueError as e:
            print(f"Error: {e}")

    def plot_categorical_distribution(self,max_un=50, max_col=10):
        categorical_columns = self.df.select_dtypes(include=['object','category']).columns
        reduced_columns = [col for col in categorical_columns if self.df[col].nunique() <= max_un]
        if True: #In the past there was an option implementation for individual or joined plots.
            if len(reduced_columns) > 0:
                reduced_columns = reduced_columns[:max_col]
                num_columns = len(reduced_columns)
                num_rows = math.ceil(num_columns/3)
                fig, axs = plt.subplots(num_rows,3,figsize=(15,5* num_rows))

                for idx, ax in enumerate(axs.flat):
                    if idx < num_columns:
                        col = reduced_columns[idx]
                        cant = self.df[col].value_counts()
                        sns.countplot(data=self.df,y=col,ax=ax,order=cant.index)
                        ax.set_title(f"Distribution of {col}")
                    else:
                        fig.delaxes(ax)
                plt.tight_layout()
                plt.show()
            else:
                print("There are no categorical values in the dataset")

    def plot_boxplots(self):
        df_numerical = self.df.select_dtypes(include=[np.number]).columns
        if len(df_numerical) > 0:
            plt.figure(figsize=(10, 5))
            self.df[df_numerical].boxplot()
            plt.title("Box Plot of Numerical Values")
            plt.xticks(rotation=10)
            plt.show()

    def plot_correlation_matrix(self):
        try:
            df_numerical = self.df.select_dtypes(include=[np.number]).columns
            corr_64 = self.df[df_numerical].corr()
            plt.figure(figsize=(12, 6))
            sns.heatmap(corr_64, annot=True)
            plt.title("Correlation of Numerical Values")
            plt.xticks(rotation=15)
            plt.show()
        except ValueError:
            print("Error: No numerical values found in the dataset")

    '''Specific data analysis'''
    def age_analysis(self):
        df_age = self.df.iloc[:, [0, 9, 10,15]]
        df_age = df_age.copy()
        df_age.loc[:, "month_at_diagnosis"] = df_age.iloc[:, 0] * 12

        # Plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 2)
        custom_palette = {'Living': 'green', 'Died of Disease': 'red','Died of Other Causes':'#FFFFFF'}
        sns.kdeplot(data=df_age, y='overall_survival_months', x='age_at_diagnosis', hue='death_from_cancer',
                        palette=custom_palette)
        plt.title("Age at diagnosis vs. Overall Survival")

        plt.subplot(1, 3, 1)
        df_survived = df_age[self.df['overall_survival'] == 1]
        sns.kdeplot(data=df_age, y=df_survived['overall_survival_months'], x='age_at_diagnosis', color='green')
        plt.title("Survived")

        plt.subplot(1, 3, 3)
        df_dead = df_age[self.df['overall_survival'] == 0]
        sns.kdeplot(data=df_age, y=df_dead['overall_survival_months'], x='age_at_diagnosis', color='red')
        plt.title("Dead")
        plt.show()

    def cancer_treatment(self):
        df_cancer = self.df.iloc[:, [3, 5, 9, 10, 12, 15]]
        '''Interesting for linear regresion'''
        custom_palette = {'Living': 'green', 'Died of Disease': 'red', 'Died of Other Causes': 'lightgrey'}
        # Plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        '''Chemotherapy'''
        sns.barplot(data=df_cancer, x='chemotherapy', y='overall_survival_months', hue='death_from_cancer',
                    palette=custom_palette)
        plt.title("Chemotherapy")
        '''Hormone therapy'''
        plt.subplot(1, 3, 2)
        sns.barplot(data=df_cancer, x='hormone_therapy', y='overall_survival_months', hue='death_from_cancer',
                    palette=custom_palette)
        plt.title("Hormone therapy")
        '''Radio Therapy'''
        plt.subplot(1, 3, 3)
        sns.barplot(data=df_cancer, x='radio_therapy', y='overall_survival_months', hue='death_from_cancer',
                    palette=custom_palette)
        plt.title("Radio Therapy")
        plt.suptitle("Cancer Treatment")
        plt.show()

        '''Deads Specific'''
        conditions = ['chemotherapy', 'hormone_therapy', 'radio_therapy']
        fig, axes = plt.subplots(nrows=len(conditions), ncols=2, figsize=(15, 5 * len(conditions)))
        fig.suptitle("Deceased % Comparison Between Treatments")

        for i,treatment in enumerate(conditions):
            df_treat_1 = df_cancer[(df_cancer[treatment] == 1) & (df_cancer['death_from_cancer'] == 'Died of Disease') |
                                   (df_cancer[treatment] == 1) & (df_cancer['death_from_cancer'] == 'Living')][
                'death_from_cancer'].value_counts()
            df_treat_1_est = df_treat_1 / df_treat_1.sum()

            df_treat_0 = df_cancer[(df_cancer[treatment] == 0) & (df_cancer['death_from_cancer'] == 'Died of Disease') |
                                   (df_cancer[treatment] == 0) & (df_cancer['death_from_cancer'] == 'Living')][
                'death_from_cancer'].value_counts()
            df_treat_0_est = df_treat_0 / df_treat_0.sum()
            print(f"\nDeath from cancer With {treatment}: n={df_treat_1.sum()}\n{df_treat_1_est}\n")
            print(f"Death from cancer Without {treatment}:n={df_treat_0.sum()}\n{df_treat_0_est}\n")

            sns.barplot(x=df_treat_1_est.index, y=df_treat_1_est.values, ax=axes[i, 0], color='#2196F3')
            axes[i, 0].set_title(f"(+) {treatment}")
            axes[i, 0].set_xlabel("")
            axes[i, 0].set_ylabel("Percentage")

            sns.barplot(x=df_treat_0_est.index, y=df_treat_0_est.values, ax=axes[i, 1], color='gray')
            axes[i, 1].set_title(f"(-) {treatment} ")
            axes[i, 1].set_xlabel("")
            axes[i, 1].set_ylabel("Percentage")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def tumor_stage_distribution(self):
        '''Tumor stage distribution'''
        df_tumor_stage = self.df['tumor_stage'].dropna().value_counts()
        print(f'\nTumor stage distribution among the pacients:\n{df_tumor_stage}\n')
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_tumor_stage)
        plt.title('Tumor stage distribution')
        plt.grid(axis='y')
        plt.show()

    def survival_analysis(self):
        df_survival_analysis = self.df[(self.df['death_from_cancer'] == 'Died of Disease') |
                                                  (self.df['death_from_cancer'] == 'Living')]
        '''Overall Survival vs Tumor Size'''
        df_size_clean = df_survival_analysis.dropna(subset=['tumor_size', 'death_from_cancer'])
        plt.figure(figsize=(16, 9))
        plt.suptitle("Survival Analysis")
        plt.subplot(2, 2, 1)
        custom_palette = {'Living': 'green', 'Died of Disease': 'red'}
        sns.boxplot(data=df_size_clean, x='overall_survival', y='tumor_size', hue='death_from_cancer',
                    palette=custom_palette)
        plt.title('Overall Survival vs Tumor Size\nOverall Survival')
        plt.xlabel('')

        '''Survival and Tumor size vs primary tumor laterality'''
        df_latearlity_clean = df_survival_analysis.dropna(
            subset=['tumor_size', 'primary_tumor_laterality', 'death_from_cancer'])

        plt.subplot(2, 2, 2)
        sns.boxplot(data=df_latearlity_clean, x='death_from_cancer', y='tumor_size', hue='primary_tumor_laterality',
                    palette='viridis')
        plt.title('Survival vs Tumor Size and laterality\nDeath')
        plt.xlabel('')


        '''Post-Chemotherapy Cellularity Status and Survival Time'''
        df_celularity_clean = df_survival_analysis.dropna(subset=['cellularity', 'death_from_cancer'])

        plt.subplot(2, 2, 3)
        sns.boxplot(data=df_celularity_clean, y='overall_survival_months', x='cellularity', hue='death_from_cancer',
                    palette=custom_palette, legend=False)
        plt.title('Post-Chemotherapy Cellularity Status and Survival Time')


        '''Overall Survival vs nottingham_prognostic_index'''
        df_nottin_clean = df_survival_analysis.dropna(subset=['death_from_cancer','nottingham_prognostic_index'])
        plt.suptitle("Survival Analysis")
        plt.subplot(2, 2, 4)
        custom_palette = {'Living': 'green', 'Died of Disease': 'red'}
        sns.boxplot(data=df_nottin_clean, x='overall_survival', y='nottingham_prognostic_index', hue='death_from_cancer',
                    palette=custom_palette,legend=False)
        plt.title(f'Overall Survival vs Nottingham Prognostic Index')
        plt.show()



    def hormonal_analysis(self):
        df_hormonal_analysis = self.df[(self.df['death_from_cancer'] == 'Died of Disease') |
                                                  (self.df['death_from_cancer'] == 'Living')]
        plt.figure(figsize=(10, 5))
        custom_palette = {'Positive': '#FFC0CB', 'Negative': '#ADD8E6'}
        sns.violinplot(data=df_hormonal_analysis, x='pr_status', y='tumor_size', hue='er_status',
                       palette=custom_palette)
        plt.title('Hormonal Receptor Status vs. Tumor Size')
        plt.show()

        '''Dead specific'''
        conditions = ['pr_status', 'er_status']
        fig, axes = plt.subplots(nrows=len(conditions), ncols=2, figsize=(15, 5 * len(conditions)))
        fig.suptitle("Deceased % Comparison Between Receptor Status")

        for i, treatment in enumerate(conditions):
            df_treat_1 = self.df[(self.df[treatment] == 'Positive') & (
                        self.df['death_from_cancer'] == 'Died of Disease') |
                                            (self.df[treatment] == 'Positive') & (
                                                        self.df['death_from_cancer'] == 'Living')][
                'death_from_cancer'].value_counts()
            df_treat_1_est = df_treat_1 / df_treat_1.sum()

            df_treat_0 = self.df[(self.df[treatment] == 'Negative') & (
                        self.df['death_from_cancer'] == 'Died of Disease') |
                                            (self.df[treatment] == 'Negative') & (
                                                        self.df['death_from_cancer'] == 'Living')][
                'death_from_cancer'].value_counts()
            df_treat_0_est = df_treat_0 / df_treat_0.sum()

            print(f"\nDeath from cancer With {treatment}: n={df_treat_1.sum()}\n{df_treat_1_est}\n")
            print(f"Death from cancer Without {treatment}:n={df_treat_0.sum()}\n{df_treat_0_est}\n")

            sns.barplot(x=df_treat_1_est.index, y=df_treat_1_est.values, ax=axes[i, 0], color='#FFC0CB')
            axes[i, 0].set_title(f"Positive {treatment} Receptor")
            axes[i, 0].set_xlabel("")
            axes[i, 0].set_ylabel("Percentage")

            sns.barplot(x=df_treat_0_est.index, y=df_treat_0_est.values, ax=axes[i, 1], color='#ADD8E6')
            axes[i, 1].set_title(f"Negative {treatment} Receptor")
            axes[i, 1].set_xlabel("")
            axes[i, 1].set_ylabel("Percentage")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
def main():
    #Dataset
    dataframe = pd.read_csv("METABRIC_RNA_Mutation.csv")
    dataframe_filteres = dataframe.iloc[:,[1,3,5,6,10,15,18,20,21,23,24,25,26,28,29,30]]

    #Data Treatment:
    breast_cancer = EDA(dataframe_filteres, 'pd')
    breast_cancer.show_columns_types()
    breast_cancer.show_first_last_row()
    breast_cancer.detect_duplicates()
    breast_cancer.missing_values()

    #EDA:
    breast_cancer.calculate_basic_values()
    breast_cancer.plot_categorical_distribution()
    breast_cancer.plot_boxplots()
    breast_cancer.plot_correlation_matrix()

    #Specific analysis:
    breast_cancer.age_analysis()
    breast_cancer.cancer_treatment()
    breast_cancer.tumor_stage_distribution()
    breast_cancer.survival_analysis()
    breast_cancer.hormonal_analysis()
    print(f'{"-"*5} Authority Santiago Britez{"-"*5}')
if __name__ == "__main__":
    main()