#!/usr/bin/env python
# coding: utf-8

# In[10]:


import openml
import pandas as pd
from sklearn.model_selection import train_test_split

domain_source = openml.datasets.get_dataset(1067)

X_source, y_source,_,_= domain_source.get_data(target=domain_source.default_target_attribute)
y_source = [1 if i else 0 for i in y_source]

# Convert X_source and y_source to DataFrames
df_source = pd.DataFrame(X_source)
df_source['Target'] = y_source


# In[11]:


df_source


# In[12]:


target_source = openml.datasets.get_dataset(1063)

X_target, y_target, _ , _  = target_source.get_data(target=target_source.default_target_attribute)

y_target = [1 if value.lower() == 'yes' else 0 for value in y_target]  

X_target.rename(columns={'lOCodeAndComment': 'locCodeAndComment'}, inplace=True)

# Convert X_source and y_source to DataFrames
df_target = pd.DataFrame(X_target)
df_target['Target'] = y_target

X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(df_target.drop(columns='Target'),df_target['Target'], test_size=0.3, random_state=42)


# In[13]:


df_target


# In[14]:


import matplotlib.pyplot as plt
import pandas as pd

# Count the number of columns in each dataset
num_columns_df1 = len(X_source.columns)
num_columns_df2 = len(X_target.columns)

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


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

missing_values_source = df_source.isnull().sum()

missing_values_target = df_target.isnull().sum()

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

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


# In[16]:


import matplotlib.pyplot as plt
import pandas as pd

# Count the number of instances in each dataset
num_instances_df1 = len(X_source)
num_instances_df2 = len(X_target)

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
plt.title('Fig 2: Number of Instances in Each Dataset', fontsize=14)

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


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the columns of interest and add the target feature to the list
columns_of_interest_source = df_source.columns.tolist() 
columns_of_interest_target = df_target.columns.tolist() 

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# Plot for Source domain
df_interest_source = df_source[columns_of_interest_source]
correlation_matrix_source = df_interest_source.corr()
sns.heatmap(correlation_matrix_source[['Target']], annot=True, cmap='coolwarm', linewidths=0.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap for Source domain with Target Feature', fontsize=14)

# Plot for Target domain
df_interest_target = df_target[columns_of_interest_target]
correlation_matrix_target = df_interest_target.corr()
sns.heatmap(correlation_matrix_target[['Target']], annot=True, cmap='coolwarm', linewidths=0.5, ax=axes[1])
axes[1].set_title('Correlation Heatmap for Target domain with Target Feature', fontsize=14)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[19]:


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import itertools
import operator
import numpy as np
import random
from deap import base, creator, tools, gp, algorithms
from sklearn.metrics import accuracy_score
import math


# In[20]:


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def Supervised_methods():
    # Decision Tree
    clf_dt = tree.DecisionTreeClassifier()
    clf_dt.fit(X_train_target, y_train_target)
    y_pred_dt = clf_dt.predict(X_test_target)
    
    accuracy_dt = accuracy_score(y_test_target, y_pred_dt)
    precision_dt = precision_score(y_test_target, y_pred_dt, pos_label=1)
    recall_dt = recall_score(y_test_target, y_pred_dt, pos_label=1)
    f1_dt = f1_score(y_test_target, y_pred_dt, pos_label=1)
    auc_roc_dt = roc_auc_score(y_test_target, clf_dt.predict_proba(X_test_target)[:, 1])

    # Random Forest
    clf_rf = RandomForestClassifier(max_depth=10, random_state=0)
    clf_rf.fit(X_train_target, y_train_target)
    y_pred_rf = clf_rf.predict(X_test_target)
    
    accuracy_rf = accuracy_score(y_test_target, y_pred_rf)
    precision_rf = precision_score(y_test_target, y_pred_rf, pos_label=1)
    recall_rf = recall_score(y_test_target, y_pred_rf, pos_label=1)
    f1_rf = f1_score(y_test_target, y_pred_rf, pos_label=1)
    auc_roc_rf = roc_auc_score(y_test_target, clf_rf.predict_proba(X_test_target)[:, 1])

    # K-Nearest Neighbors
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X_train_target, y_train_target)
    y_pred_knn = knn_clf.predict(X_test_target)
    
    accuracy_knn = accuracy_score(y_test_target, y_pred_knn)
    precision_knn = precision_score(y_test_target, y_pred_knn, pos_label=1)
    recall_knn = recall_score(y_test_target, y_pred_knn, pos_label=1)
    f1_knn = f1_score(y_test_target, y_pred_knn, pos_label=1)
    auc_roc_knn = roc_auc_score(y_test_target, knn_clf.predict_proba(X_test_target)[:, 1])

    return {
        'Decision Tree': {'Accuracy': accuracy_dt, 'Precision': precision_dt, 'Recall': recall_dt, 'F1 Score': f1_dt, 'AUC-ROC': auc_roc_dt},
        'Random Forest': {'Accuracy': accuracy_rf, 'Precision': precision_rf, 'Recall': recall_rf, 'F1 Score': f1_rf, 'AUC-ROC': auc_roc_rf},
        'K-Nearest Neighbors': {'Accuracy': accuracy_knn, 'Precision': precision_knn, 'Recall': recall_knn, 'F1 Score': f1_knn, 'AUC-ROC': auc_roc_knn}
    }


# In[39]:


def GeneticProgramming_with_transfer(no_features):
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

       def evaluation_function(individual, data, labels, toolbox):
           func = toolbox.compile(expr=individual)
           correct_predictions = 0

           for row, label in zip(data, labels):
               try:
                   output = func(*row[:no_features+1])
                   prediction = output > 0.5
                   if prediction == bool(label):
                       correct_predictions += 1
               except Exception as e:
                   print(f"Error evaluating individual: {e}")

           accuracy = correct_predictions / len(labels)
           return accuracy,
       
       def transfer_individuals(target_population, source_hof):
           """
           Transfer a percentage of individuals from the Hall of Fame of the source population
           to the target population.
           """
           for i in range(len(source_hof)):
               target_population[i] = source_hof[i]
           return target_population
       
       def main():
           
           # Initialize the toolbox
           toolbox = base.Toolbox()

           # Common registration for both source and target populations
           toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=5)
           toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
           toolbox.register("population", tools.initRepeat, list, toolbox.individual)
           toolbox.register("compile", gp.compile, pset=pset)

           # Source population registration
           toolbox_source = toolbox.clone(toolbox)
           population_source = toolbox_source.population(n=1500)
           toolbox_source.register("evaluate", evaluation_function, data=X_source.drop(columns='Target').values.tolist(), labels=y_source, toolbox = toolbox_source)
           toolbox_source.register("select", lambda population_source, k: tools.selBest(population_source, k=150))
           toolbox_source.register("mate", gp.cxOnePoint)
           toolbox_source.register("expr_mut", gp.genFull, min_=3, max_=5)
           toolbox_source.register("mutate", gp.mutUniform, expr=toolbox_source.expr_mut, pset=pset)
           toolbox_source.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
           toolbox_source.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

           # Target population registration
           toolbox_target = toolbox.clone(toolbox)
           population_target = toolbox_source.population(n=1500)
           toolbox_target.register("evaluate", evaluation_function, data=X_train_target.values.tolist(), labels=y_train_target, toolbox = toolbox_target)
           toolbox_target.register("select", lambda population_target, k: tools.selBest(population_target, k=150))
           toolbox_target.register("mate", gp.cxOnePoint)
           toolbox_target.register("expr_mut", gp.genFull, min_=3, max_=5)
           toolbox_target.register("mutate", gp.mutUniform, expr=toolbox_target.expr_mut, pset=pset)
           toolbox_target.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
           toolbox_target.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
           
           # Source population
           hof_source = tools.HallOfFame(100)
           stats_source = tools.Statistics(lambda ind: ind.fitness.values)
           stats_source.register("avg", np.mean)
           stats_source.register("std", np.std)
           stats_source.register("min", np.min)
           stats_source.register("max", np.max)

           # Run the evolutionary algorithm for the source population
           for gen in range(20):
               algorithms.eaSimple(population_source, toolbox_source, cxpb=0.9, mutpb=0.4,
                                    ngen=1, stats=stats_source, halloffame=hof_source, verbose=True)

               # Extract the best individuals from the current generation
               current_best_individuals = tools.selAutomaticEpsilonLexicase(population_source, 10)

               # Combine the best individuals from the current and previous runs
               combined_elite = current_best_individuals + hof_source.items

               # Keep only the top 10% as elite individuals
               num_elite = int(0.1 * len(combined_elite))
               combined_elite = tools.sortNondominated(combined_elite, len(combined_elite), first_front_only=True)[0][:num_elite]

               # Replace the current population with the combined elite individuals
               population_source[:] = combined_elite

           population_target = transfer_individuals(population_target,hof_source)
           
           hof_target = tools.HallOfFame(100)
           stats_target = tools.Statistics(lambda ind: ind.fitness.values)
           stats_target.register("avg", np.mean)
           stats_target.register("std", np.std)
           stats_target.register("min", np.min)
           stats_target.register("max", np.max)
           
           # Run the evolutionary algorithm for the target population
           for gen in range(20):
               algorithms.eaSimple(population_target, toolbox_target, cxpb=0.9, mutpb=0.4,
                                    ngen=1, stats=stats_target, halloffame=hof_target, verbose=True)

               # Extract the best individuals from the current generation
               current_best_individuals = tools.selAutomaticEpsilonLexicase(population_target, 10)

               # Combine the best individuals from the current and previous runs
               combined_elite = current_best_individuals + hof_target.items

               # Keep only the top 10% as elite individuals
               num_elite = int(0.1 * len(combined_elite))
               combined_elite = tools.sortNondominated(combined_elite, len(combined_elite), first_front_only=True)[0][:num_elite]

               # Replace the current population with the combined elite individuals
               population_target[:] = combined_elite
           
           return hof_target
   
       if __name__ == "__main__":
           hof_second = main()
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
           y_pred_strings = np.array([1 if pred == 1 else 0 for pred in y_pred_binary])

           # Calculate precision, recall, and F1-score using the binary predictions
           precision = precision_score(y_test_target, y_pred_strings, pos_label=1)
           recall = recall_score(y_test_target, y_pred_strings, pos_label=1)
           f1 = f1_score(y_test_target, y_pred_strings, pos_label=1)
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


# In[40]:


no_features_value = 21
GPwith = GeneticProgramming_with_transfer(no_features_value)


# In[45]:


def GeneticProgramming_without_transfer(no_features):
        #Create GP tree
        input_types = [float] * no_features
        pset = gp.PrimitiveSetTyped("first", input_types, float)
        for i in range(10):
            pset.renameArguments(**{f"ARG{i}": f"IN{i}"})
    
        def protected_divine(x, y):
            return 1 if y == 0 else x / y

        # Add all operators and protected_divine to the primitive set
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(protected_divine, [float, float], float)
        pset.addPrimitive(operator.sub, [float,float], float)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Use FitnessMin for minimization problems
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox_second = base.Toolbox()
        def evaluation_function(individual, data, labels):
            func = toolbox_second.compile(expr=individual)
            correct_predictions = 0

            for row, label in zip(data, labels):
                try:
                    output = func(*row[:no_features+1])
                    prediction = output > 0.5
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

        toolbox_second.register("evaluate", evaluation_function, data=X_train_target.values.tolist(), labels=y_train_target)
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
            

            population_target = toolbox_second.population(n=1500)
            hof_target = tools.HallOfFame(20)


            population, logbook = algorithms.eaSimple(population_target, toolbox_second, cxpb=0.5, mutpb=0.5,
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
            y_pred_strings = np.array([1 if pred == 1 else 0 for pred in y_pred_binary])

            # Calculate precision, recall, and F1-score using the binary predictions
            precision = precision_score(y_test_target, y_pred_strings, pos_label=1)
            recall = recall_score(y_test_target, y_pred_strings, pos_label=1)
            f1 = f1_score(y_test_target, y_pred_strings, pos_label=1)
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
        'Genetic Programming without transfer learning': {'Accuracy': best_accuracy, 'Precision': best_precision, 'Recall': best_recall, 'F1 Score': best_f1, 'AUC-ROC': best_auc_roc},
        }

        # Call the function and store the results
        results = Supervised_methods()

        # Display the results
        for model, metrics in results.items():
            print(f"\n{model} Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")


# In[46]:


no_features_value = 21
GPwithout = GeneticProgramming_without_transfer(no_features_value)
GPwithout


# In[50]:


import numpy as np
import matplotlib.pyplot as plt

results = Supervised_methods()
results.update(GPwith)
results.update(GPwithout)

models = list(results.keys())
metrics = list(results[models[0]].keys())
num_models = len(models)

bar_width = 0.1  # Width of each bar
index = np.arange(len(models))  # Index for each group

plt.figure(figsize=(15, 10))

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
plt.title('Fig3: Performance Metrics Comparison')

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




