# %% [markdown]
# # performance analysis 

# %% [markdown]
# dependencies

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# %% [markdown]
# training history

# %%
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# Confusion Matrix

# %%
def plot_cf_matrix(model, test_generator):
    test_generator.reset()
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    class_names = list(test_generator.class_indices.keys())

    print('Confusion Matrix')
    conf_metrix = confusion_matrix(test_generator.classes, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_metrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

# %% [markdown]
# Misclassifications plot

# %%
def plot_misclassifications(model, test_generator):
    test_generator.reset()
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    class_names = list(test_generator.class_indices.keys())
    conf_metrix = confusion_matrix(test_generator.classes, y_pred)
    
    print('Misclassifications')
    misclassifications = np.sum(conf_metrix, axis=1) - np.diag(conf_metrix)
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, misclassifications, color='red')
    plt.xlabel('Class Labels')
    plt.ylabel('Number of Misclassifications')
    plt.title('Misclassifications for Each Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# Classification Reports

# %%
def reports(model, test_generator):
    test_generator.reset()
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    class_names = list(test_generator.class_indices.keys())

    print('Classification Reports')
    print(classification_report(test_generator.classes, y_pred, target_names=class_names))

    test_loss, test_acc = model.evaluate(test_generator)
    print("Test Accuracy:", test_acc)
    print("Test Loss:", test_loss)

# %% [markdown]
# AUC-ROC

# %%
def plot_roc_curve_with_auc(model, generator):
    class_labels = list(generator.class_indices.keys())
    num_classes = len(class_labels)
    y_true = generator.classes  
    y_pred = model.predict(generator)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    auc_scores = roc_auc_score(y_true_bin, y_pred, multi_class='ovr', average=None)
    
    fpr = dict()
    tpr = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    
    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_labels[i]} (AUC = {auc_scores[i]:.2f})')
    
    # Plot the diagonal line for a random classifier
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# %% [markdown]
# measurements

# %%
def all_measurement(history,model,generator):
    plot_training_history(history)
    plot_cf_matrix(model,generator)
    reports(model,generator)
    plot_roc_curve_with_auc(model,generator)
    plot_misclassifications(model,generator)


