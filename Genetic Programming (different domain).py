#!/usr/bin/env python
# coding: utf-8

# In[257]:


import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import operator
import numpy as np
import random
from deap import base, creator, tools, gp, algorithms
from sklearn.metrics import accuracy_score
import math
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn import tree


# In[258]:


Source_domain = pd.read_csv("heart.csv")


# In[259]:


target_domain = pd.read_csv("healthcare-dataset-stroke-data.csv")


# In[260]:


import matplotlib.pyplot as plt
import pandas as pd

# Count the number of columns in each dataset
num_columns_df1 = len(Source_domain.columns)
num_columns_df2 = len(target_domain.columns)

# Create a bar chart with styling
bar_width = 0.35  # Adjust the width of the bars as needed

fig, ax = plt.subplots()

bars1 = ax.bar(1, height=num_columns_df1, width=bar_width, label='Source Domain', color='skyblue', edgecolor='black')
bars2 = ax.bar(2, height=num_columns_df2, width=bar_width, label='Target Domain', color='lightcoral', edgecolor='black', align='edge')

# Add the number of columns on top of each bar
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Set labels and title
plt.xlabel('Domains', fontsize=12)
plt.ylabel('Number of Columns', fontsize=12)
plt.title('Fig1 : Number of Columns in Each Dataset', fontsize=14)

# Set x-axis ticks and remove x-axis ticks
ax.set_xticks([1, 2])
ax.set_xticklabels(['Source Domain', 'Target Domain'])
ax.tick_params(axis='x', which='both', bottom=False)

# Add legend
ax.legend()

# Add grid lines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add background color
ax.set_facecolor('#F5F5F5')

# Show the plot
plt.show()


# In[261]:


import matplotlib.pyplot as plt
import pandas as pd

# Count the number of numerical and categorical features in each dataset
num_numerical_df1 = len(Source_domain.select_dtypes(include='number').columns)
num_categorical_df1 = len(Source_domain.select_dtypes(exclude='number').columns)

num_numerical_df2 = len(target_domain.select_dtypes(include='number').columns)
num_categorical_df2 = len(target_domain.select_dtypes(exclude='number').columns)

# Create a bar chart with styling
bar_width = 0.35  # Adjust the width of the bars as needed
bar_distance = 0.4  # Adjust the distance between numerical and categorical bars

fig, ax = plt.subplots()

# Plotting the number of numerical features
bars1 = ax.bar(1 - bar_distance/2, height=num_numerical_df1, width=bar_width, label='Numerical (Source)', color='skyblue', edgecolor='black')
bars2 = ax.bar(1 + bar_distance/2, height=num_numerical_df2, width=bar_width, label='Numerical (Target)', color='lightcoral', edgecolor='black')

# Plotting the number of categorical features
bars3 = ax.bar(2 - bar_distance/2, height=num_categorical_df1, width=bar_width, label='Categorical (Source)', color='lightgreen', edgecolor='black')
bars4 = ax.bar(2 + bar_distance/2, height=num_categorical_df2, width=bar_width, label='Categorical (Target)', color='lightpink', edgecolor='black')

# Add the number of columns on top of each bar
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Set labels and title
plt.xlabel('Features', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Fig 2: Number of Numerical and Categorical Features in Each Dataset', fontsize=14)

# Set x-axis ticks and remove x-axis ticks
ax.set_xticks([1, 2])
ax.set_xticklabels(['Source Domain', 'Target Domain'])
ax.tick_params(axis='x', which='both', bottom=False)

# Add legend
ax.legend()

# Add grid lines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add background color
ax.set_facecolor('#F5F5F5')

# Show the plot
plt.show()


# In[262]:


import matplotlib.pyplot as plt
import pandas as pd

# Count the number of instances in each dataset
num_instances_df1 =  len(Source_domain)
num_instances_df2 = len(target_domain)

# Create a bar chart with styling
bar_width = 0.35  # Adjust the width of the bars as needed

fig, ax = plt.subplots()

bars1 = ax.bar(1, height=num_instances_df1, width=bar_width, label='Source Domain', color='skyblue', edgecolor='black')
bars2 = ax.bar(2, height=num_instances_df2, width=bar_width, label='Target Domain', color='lightcoral', edgecolor='black', align='edge')

# Add the number of instances on top of each bar
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Set labels and title
plt.xlabel('Domains', fontsize=12)
plt.ylabel('Number of Instances', fontsize=12)
plt.title('Fig 3: Number of Instances in Each Dataset', fontsize=14)

# Set x-axis ticks and remove x-axis ticks
ax.set_xticks([1, 2])
ax.set_xticklabels(['Source Domain', 'Target Domain'])
ax.tick_params(axis='x', which='both', bottom=False)

# Add legend
ax.legend()

# Add grid lines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add background color
ax.set_facecolor('#F5F5F5')

# Show the plot
plt.show()


# In[263]:


import matplotlib.pyplot as plt
import pandas as pd

# Count the number of instances in each dataset
num_instances_df1 =  len(Source_domain)
num_instances_df2 = len(target_domain)

# Create a bar chart with styling
bar_width = 0.35  # Adjust the width of the bars as needed

fig, ax = plt.subplots()

bars1 = ax.bar(1, height=num_instances_df1, width=bar_width, label='Source Domain', color='skyblue', edgecolor='black')
bars2 = ax.bar(2, height=num_instances_df2, width=bar_width, label='Target Domain', color='lightcoral', edgecolor='black', align='edge')

# Add the number of instances on top of each bar
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Set labels and title
plt.xlabel('Domains', fontsize=12)
plt.ylabel('Number of Instances', fontsize=12)
plt.title('Fig 3: Number of Instances in Each Dataset', fontsize=14)

# Set x-axis ticks and remove x-axis ticks
ax.set_xticks([1, 2])
ax.set_xticklabels(['Source Domain', 'Target Domain'])
ax.tick_params(axis='x', which='both', bottom=False)

# Add legend
ax.legend()

# Add grid lines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add background color
ax.set_facecolor('#F5F5F5')

# Show the plot
plt.show()


# In[281]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

missing_values_source = Source_domain.isnull().sum()

missing_values_target = target_domain.isnull().sum()

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot for df_source
axes[0].bar(missing_values_source.index, missing_values_source, color='skyblue', edgecolor='black')
axes[0].set_title('Number of Missing Values in Each Column (Source Domain)')
axes[0].set_xlabel('Columns')
axes[0].set_ylabel('Number of Missing Values')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels

# Plot for df_target
axes[1].bar(missing_values_target.index, missing_values_target, color='skyblue', edgecolor='black')
axes[1].set_title('Number of Missing Values in Each Column (Target Domain)')
axes[1].set_xlabel('Columns')
axes[1].set_ylabel('Number of Missing Values')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[265]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Source_domain['Sex'] = label_encoder.fit_transform(Source_domain['Sex'])
Source_domain['ChestPainType'] = label_encoder.fit_transform(Source_domain['ChestPainType'])
Source_domain['RestingECG'] = label_encoder.fit_transform(Source_domain['RestingECG'])
Source_domain['ExerciseAngina'] = label_encoder.fit_transform(Source_domain['ExerciseAngina'])
Source_domain['ST_Slope'] = label_encoder.fit_transform(Source_domain['ST_Slope'])


# In[266]:


X_source= Source_domain.drop(columns='HeartDisease')

y_source = Source_domain['HeartDisease']

y_source = ['Yes' if i == 1 else 'no' for i in y_source]


# In[268]:


target_domain['gender'] = label_encoder.fit_transform(target_domain['gender'])
target_domain['ever_married'] = label_encoder.fit_transform(target_domain['ever_married'])
target_domain['work_type'] = label_encoder.fit_transform(target_domain['work_type'])
target_domain['Residence_type'] = label_encoder.fit_transform(target_domain['Residence_type'])
target_domain['smoking_status'] = label_encoder.fit_transform(target_domain['smoking_status'])
target_domain['bmi'] = label_encoder.fit_transform(target_domain['bmi'])

target_domain['stroke'] = ['Yes' if i == 1 else 'no' for i in target_domain['stroke']]

# Separate features and labels
X = target_domain.drop('stroke', axis=1)
y = target_domain['stroke']

# Split the data into training and testing sets
X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply random undersampling
undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train_target, y_train_target)
X_test_resampled, y_test_resampled = undersampler.fit_resample(X_test_target, y_test_target)


# In[270]:


import matplotlib.pyplot as plt
import pandas as pd

# Count the number of instances in each dataset
num_instances_df1 = len(target_domain[target_domain['stroke'] == "Yes"])
num_instances_df2 =  len(target_domain[target_domain['stroke'] == "no"])

# Create a bar chart with styling
bar_width = 0.35  # Adjust the width of the bars as needed

fig, ax = plt.subplots()

bars1 = ax.bar(1, height=num_instances_df1, width=bar_width, label='True Label', color='skyblue', edgecolor='black')
bars2 = ax.bar(2, height=num_instances_df2, width=bar_width, label='False Label', color='lightcoral', edgecolor='black', align='edge')

# Add the number of instances on top of each bar
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Set labels and title
plt.xlabel('Domains', fontsize=12)
plt.ylabel('Number of Instances', fontsize=12)
plt.title('Fig 4: The number of instances for each label in the target domain', fontsize=14)

# Set x-axis ticks and remove x-axis ticks
ax.set_xticks([1, 2])
ax.set_xticklabels(['Source Domain', 'Target Domain'])
ax.tick_params(axis='x', which='both', bottom=False)

# Add legend
ax.legend()

# Add grid lines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add background color
ax.set_facecolor('#F5F5F5')

# Show the plot
plt.show()


# In[277]:


def Supervised_methods():
    # Decision Tree
    clf_dt = DecisionTreeClassifier()
    clf_dt.fit(X_resampled, y_resampled)
    y_pred_dt = clf_dt.predict(X_test_target)
    
    accuracy_dt = accuracy_score(y_test_target, y_pred_dt)
    precision_dt = precision_score(y_test_target, y_pred_dt, pos_label='Yes')
    recall_dt = recall_score(y_test_target, y_pred_dt, pos_label='Yes')
    f1_dt = f1_score(y_test_target, y_pred_dt, pos_label='Yes')
    auc_roc_dt = roc_auc_score(y_test_target, clf_dt.predict_proba(X_test_target)[:, 1])

    # Random Forest
    clf_rf = RandomForestClassifier(max_depth=10, random_state=0)
    clf_rf.fit(X_resampled, y_resampled)
    y_pred_rf = clf_rf.predict(X_test_target)
    
    accuracy_rf = accuracy_score(y_test_target, y_pred_rf)
    precision_rf = precision_score(y_test_target, y_pred_rf, pos_label='Yes')
    recall_rf = recall_score(y_test_target, y_pred_rf, pos_label='Yes')
    f1_rf = f1_score(y_test_target, y_pred_rf, pos_label='Yes')
    auc_roc_rf = roc_auc_score(y_test_target, clf_rf.predict_proba(X_test_target)[:, 1])

    # K-Nearest Neighbors
    knn_clf = KNeighborsClassifier(n_neighbors=12)
    knn_clf.fit(X_resampled, y_resampled)
    y_pred_knn = knn_clf.predict(X_test_target)
    
    accuracy_knn = accuracy_score(y_test_target, y_pred_knn)
    precision_knn = precision_score(y_test_target, y_pred_knn, pos_label='Yes')
    recall_knn = recall_score(y_test_target, y_pred_knn, pos_label='Yes')
    f1_knn = f1_score(y_test_target, y_pred_knn, pos_label='Yes')
    auc_roc_knn = roc_auc_score(y_test_target, knn_clf.predict_proba(X_test_target)[:, 1])

    return {
        'Decision Tree': {'Accuracy': accuracy_dt, 'Precision': precision_dt, 'Recall': recall_dt, 'F1 Score': f1_dt, 'AUC-ROC': auc_roc_dt},
        'Random Forest': {'Accuracy': accuracy_rf, 'Precision': precision_rf, 'Recall': recall_rf, 'F1 Score': f1_rf, 'AUC-ROC': auc_roc_rf},
        'K-Nearest Neighbors': {'Accuracy': accuracy_knn, 'Precision': precision_knn, 'Recall': recall_knn, 'F1 Score': f1_knn, 'AUC-ROC': auc_roc_knn}
    }


# In[278]:


# Call the function and store the results
results = Supervised_methods()

# Display the results
for model, metrics in results.items():
    print(f"\n{model} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")


# In[229]:


def Pretrained_GP(no_features):
        #Create GP tree
        input_types = [float] * no_features
        pset = gp.PrimitiveSetTyped("first", input_types, float)
        for i in range(11):
            pset.renameArguments(**{f"ARG{i}": f"IN{i}"})
    
        def protected_divine(x, y):
            return 0 if y == 0 else x / y

        # Add all operators and protected_divine to the primitive set
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(protected_divine, [float, float], float)
        pset.addPrimitive(operator.sub, [float,float], float)
        pset.addPrimitive(math.sin, [float], float)  # Sine
        pset.addPrimitive(math.cos, [float], float)  # Cosine
        pset.addPrimitive(math.tanh, [float], float)  # Hyperbolic tangent

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox_second = base.Toolbox()
        def evaluation_function(individual, data, labels):
            func = toolbox_second.compile(expr=individual)
            correct_predictions = 0

            for row, label in zip(data, labels):
                try:
                    output = func(*row[:no_features+1])
                    prediction = output >=0
                    if prediction == bool(label):
                        correct_predictions += 1
                except Exception as e:
                    print(f"Error evaluating individual: {e}")

            accuracy = correct_predictions / len(labels)
            return accuracy,
        
        toolbox_second.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox_second.register("individual", tools.initIterate, creator.Individual, toolbox_second.expr)
        toolbox_second.register("population", tools.initRepeat, list, toolbox_second.individual)
        toolbox_second.register("compile", gp.compile, pset=pset)

        toolbox_second.register("evaluate", evaluation_function, data=X_source.values.tolist(), labels=y_source)
        toolbox_second.register("select", tools.selTournament, tournsize=15)
        toolbox_second.register("mate", gp.cxOnePoint)
        toolbox_second.register("expr_mut", gp.genFull, min_=1, max_=2)
        toolbox_second.register("mutate", gp.mutUniform, expr=toolbox_second.expr_mut, pset=pset)
        toolbox_second.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox_second.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        stats_second = tools.Statistics(lambda ind: ind.fitness.values)
        
            
        def main_second():
            """
            Create the population for the second GP with a percentage of best individuals from the
            Hall of Fame of the source population
            """
            stats_second = tools.Statistics(lambda ind: ind.fitness.values)
            stats_second.register("avg", np.mean)
            stats_second.register("std", np.std)
            stats_second.register("min", np.min)
            stats_second.register("max", np.max)
            

            population_target = toolbox_second.population(n=1200)
            hof_target = tools.HallOfFame(150)


            population, logbook = algorithms.eaSimple(population_target, toolbox_second, cxpb=0.9, mutpb=0.9,
                                                      ngen=15, stats=stats_second, halloffame=hof_target, verbose=True)

            return population, logbook, hof_target
        
        
        if __name__ == "__main__":
            _, _, hof_second = main_second()
            hof_second_dataset = list(hof_second)

        return hof_second


# In[188]:


hof = Pretrained_GP(11)


# In[241]:


def transfer_individuals(target_population, source_hof, num_target_features):
    """
    Transfer a percentage of individuals from the Hall of Fame of the source population
    to the target population. Assumes source_hof individuals have num_target_features.
    """
    for i in range(len(source_hof)):
        source_individual = source_hof[i]
        
        # Check if the source individual has the same number of features as the target
        if len(source_individual) != num_target_features:
            continue
            
        # Transfer the individual to the target population
        target_population[i] = source_individual

    return target_population

def GeneticProgramming_with_transfer(no_features):
        #Create GP tree
        input_types = [float] * no_features
        pset = gp.PrimitiveSetTyped("first", input_types, float)
        for i in range(11):
            pset.renameArguments(**{f"ARG{i}": f"IN{i}"})
    
        def protected_divine(x, y):
            return 0 if y == 0 else x / y

        # Add all operators and protected_divine to the primitive set
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(protected_divine, [float, float], float)
        pset.addPrimitive(operator.sub, [float,float], float)
        pset.addPrimitive(math.sin, [float], float)  # Sine
        pset.addPrimitive(math.cos, [float], float)  # Cosine
        pset.addPrimitive(math.tanh, [float], float)  # Hyperbolic tangent

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Use FitnessMin for minimization problems
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox_second = base.Toolbox()
        def evaluation_function(individual, data, labels):
            func = toolbox_second.compile(expr=individual)
            correct_predictions = 0

            for row, label in zip(data, labels):
                try:
                    output = func(*row[:no_features+1])
                    prediction = output >=0
                    if prediction == bool(label):
                        correct_predictions += 1
                except Exception as e:
                    print(f"Error evaluating individual: {e}")

            accuracy = correct_predictions / len(labels)
            return accuracy,
        
        toolbox_second.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox_second.register("individual", tools.initIterate, creator.Individual, toolbox_second.expr)
        toolbox_second.register("population", tools.initRepeat, list, toolbox_second.individual)
        toolbox_second.register("compile", gp.compile, pset=pset)

        
        toolbox_second.register("evaluate", evaluation_function, data=X_resampled.values.tolist(), labels=y_resampled)
        toolbox_second.register("select", tools.selTournament, tournsize=20)
        toolbox_second.register("mate", gp.cxOnePoint)
        toolbox_second.register("expr_mut", gp.genFull, min_=1, max_=2)
        toolbox_second.register("mutate", gp.mutUniform, expr=toolbox_second.expr_mut, pset=pset)
        toolbox_second.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox_second.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        stats_second = tools.Statistics(lambda ind: ind.fitness.values)
            
        def main_second():
            """
            Create the population for the second GP with a percentage of best individuals from the
            Hall of Fame of the source population
            """
            stats_second = tools.Statistics(lambda ind: ind.fitness.values)
            stats_second.register("avg", np.mean)
            stats_second.register("std", np.std)
            stats_second.register("min", np.min)
            stats_second.register("max", np.max)
            

            population_target = toolbox_second.population(n=1200)
            
            population_target = transfer_individuals(population_target, hof, no_features)
            
            hof_target = tools.HallOfFame(150)


            population, logbook = algorithms.eaSimple(population_target, toolbox_second, cxpb=0.9, mutpb=0.9,
                                                      ngen=15, stats=stats_second, halloffame=hof_target, verbose=True)

            return population, logbook, hof_target
        
        if __name__ == "__main__":
            _, _, hof_second = main_second()
            hof_second_dataset = list(hof_second)

        # Iterate through each individual in the Hall of Fame
        best_accuracy = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        best_auc_roc = 0.0
        best_pred = []

        for i, best_individual in enumerate(hof_second_dataset):
            # Compile the expression into a Python function
            compiled_function = gp.compile(expr=best_individual, pset=pset)

            # Evaluate on the test set
            y_pred_continuous = np.array([compiled_function(*row[:no_features]) for row in X_test_target.values])

            best_threshold = 0.5  # Adjust this threshold as needed
            y_pred_binary = (y_pred_continuous >= best_threshold).astype(int)

            # Convert integer labels to strings
            y_pred_strings = np.array(["Yes" if pred == 1 else "no" for pred in y_pred_binary])

            # Calculate precision, recall, and F1-score using the binary predictions
            precision = precision_score(y_test_target, y_pred_strings, pos_label='Yes')
            recall = recall_score(y_test_target, y_pred_strings, pos_label='Yes')
            f1 = f1_score(y_test_target, y_pred_strings, pos_label='Yes')
            auc_roc = roc_auc_score(y_test_target, y_pred_continuous)

            # Update best metrics if the current individual performs better
            if f1 > best_f1:
                best_f1 = f1
                best_pred = y_pred_strings
            if accuracy_score(y_test_target, y_pred_strings) > best_accuracy:
                best_accuracy = accuracy_score(y_test_target, y_pred_strings)
            if recall > best_recall:
                best_recall = recall
            if precision > best_precision:
                best_precision = precision
            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc

                
        return {
        'Genetic Programming with transfer learning': {'Accuracy': best_accuracy, 'Precision': best_precision, 'Recall': best_recall, 'F1 Score': best_f1, 'AUC-ROC': best_auc_roc},
        }

        # Call the function and store the results
        results = Supervised_methods()

        # Display the results
        for model, metrics in results.items():
            print(f"\n{model} Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")


# In[242]:


GPwith = GeneticProgramming_with_transfer(11)


# In[243]:


GPwith


# In[244]:


def GeneticProgramming_without_transfer(no_features):
        #Create GP tree
        input_types = [float] * no_features
        pset = gp.PrimitiveSetTyped("first", input_types, float)
        for i in range(10):
            pset.renameArguments(**{f"ARG{i}": f"IN{i}"})
    
        def protected_divine(x, y):
            return 0 if y == 0 else x / y

        # Add all operators and protected_divine to the primitive set
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(protected_divine, [float, float], float)
        pset.addPrimitive(operator.sub, [float,float], float)
        pset.addPrimitive(math.sin, [float], float)  # Sine
        pset.addPrimitive(math.cos, [float], float)  # Cosine
        pset.addPrimitive(math.tanh, [float], float)  # Hyperbolic tangent

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Use FitnessMin for minimization problems
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox_second = base.Toolbox()
        def evaluation_function(individual, data, labels):
            func = toolbox_second.compile(expr=individual)
            correct_predictions = 0

            for row, label in zip(data, labels):
                try:
                    output = func(*row[:no_features+1])
                    prediction = output >=0
                    if prediction == bool(label):
                        correct_predictions += 1
                except Exception as e:
                    print(f"Error evaluating individual: {e}")

            accuracy = correct_predictions / len(labels)
            return accuracy,
        
        toolbox_second.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox_second.register("individual", tools.initIterate, creator.Individual, toolbox_second.expr)
        toolbox_second.register("population", tools.initRepeat, list, toolbox_second.individual)
        toolbox_second.register("compile", gp.compile, pset=pset)

        toolbox_second.register("evaluate", evaluation_function, data=X_resampled.values.tolist(), labels=y_resampled)
        toolbox_second.register("select", tools.selTournament, tournsize=15)
        toolbox_second.register("mate", gp.cxOnePoint)
        toolbox_second.register("expr_mut", gp.genFull, min_=1, max_=2)
        toolbox_second.register("mutate", gp.mutUniform, expr=toolbox_second.expr_mut, pset=pset)
        toolbox_second.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox_second.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        stats_second = tools.Statistics(lambda ind: ind.fitness.values)
            
        def main_second():
            """
            Create the population for the second GP with a percentage of best individuals from the
            Hall of Fame of the source population
            """
            stats_second = tools.Statistics(lambda ind: ind.fitness.values)
            stats_second.register("avg", np.mean)
            stats_second.register("std", np.std)
            stats_second.register("min", np.min)
            stats_second.register("max", np.max)
            

            population_target = toolbox_second.population(n=1200)
            hof_target = tools.HallOfFame(150)


            population, logbook = algorithms.eaSimple(population_target, toolbox_second, cxpb=0.9, mutpb=0.9,
                                                      ngen=15, stats=stats_second, halloffame=hof_target, verbose=True)

            return population, logbook, hof_target
        
        if __name__ == "__main__":
            _, _, hof_second = main_second()
            hof_second_dataset = list(hof_second)

        # Iterate through each individual in the Hall of Fame
        best_accuracy = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        best_auc_roc = 0.0
        best_pred = []

        for i, best_individual in enumerate(hof_second_dataset):
            # Compile the expression into a Python function
            compiled_function = gp.compile(expr=best_individual, pset=pset)

            # Evaluate on the test set
            y_pred_continuous = np.array([compiled_function(*row[:no_features]) for row in X_test_target.values])

            best_threshold = 0.5  # Adjust this threshold as needed
            y_pred_binary = (y_pred_continuous >= best_threshold).astype(int)

            # Convert integer labels to strings
            y_pred_strings = np.array(["Yes" if pred == 1 else "no" for pred in y_pred_binary])

            # Calculate precision, recall, and F1-score using the binary predictions
            precision = precision_score(y_test_target, y_pred_strings, pos_label='Yes')
            recall = recall_score(y_test_target, y_pred_strings, pos_label='Yes')
            f1 = f1_score(y_test_target, y_pred_strings, pos_label='Yes')
            auc_roc = roc_auc_score(y_test_target, y_pred_continuous)

            # Update best metrics if the current individual performs better
            if f1 > best_f1:
                best_f1 = f1
            if accuracy_score(y_test_target, y_pred_strings) > best_accuracy:
                best_accuracy = accuracy_score(y_test_target, y_pred_strings)
                best_pred = y_pred_strings
            if recall > best_recall:
                best_recall = recall
            if precision > best_precision:
                best_precision = precision
            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc

                
        return {
        'Genetic Programming without transfer learning': {'Accuracy': best_accuracy, 'Precision': best_precision, 'Recall': best_recall, 'F1 Score': best_f1, 'AUC-ROC': best_auc_roc},
        }

        # Call the function and store the results
        results = Supervised_methods()

        # Display the results
        for model, metrics in results.items():
            print(f"\n{model} Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")


# In[245]:


GPwithout = GeneticProgramming_without_transfer(11)


# In[280]:


import numpy as np
import matplotlib.pyplot as plt

results.update(GPwith)
results.update(GPwithout)

models = list(results.keys())
metrics = list(results[models[0]].keys())
num_models = len(models)

bar_width = 0.1  # Width of each bar
index = np.arange(len(models))  # Index for each group

plt.figure(figsize=(12, 10))

# Plot each metric as a clustered column bar
colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightgoldenrodyellow', 'lightpink']
for i, metric in enumerate(metrics):
    metric_values = [results[model][metric] for model in models]
    bars = plt.bar(index + i * bar_width, metric_values, bar_width, label=metric, color=colors[i])

    # Annotate each bar with its corresponding value
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom')

plt.xlabel('Models')
plt.ylabel('Performance Metric Values')
plt.title('Fig5 : Performance Metrics Comparison')

# Rotate x-labels vertically
plt.xticks(rotation='vertical')
plt.xticks(index + (num_models - 1) * bar_width / 2, models)

plt.legend()

# Add a grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Customize the legend location
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Beautify the plot
plt.style.use('seaborn-darkgrid')

plt.show()


# In[ ]:




