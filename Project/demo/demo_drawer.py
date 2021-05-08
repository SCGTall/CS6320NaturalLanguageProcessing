# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt

files = ["static/x0.csv", "static/x2.csv", "static/x3.csv", "static/x4.csv"]
titles = ["Only m_id", "m_id and sentiment", "m_id and emotion", "m_id and topic"]

for (index, file) in enumerate(files):
    print(file)
    with open(file, 'r') as data:
        training_loss = []
        validation_loss = []
        f1_score = []
        agree_accuracy = []
        disagree_accuracy = []
        no_stance_accuracy = []
        not_relevant_accuracy = []
        total_accuracy = []
        for line in csv.DictReader(data):
            #print(line)
            training_loss.append(float(line['Training loss']))
            validation_loss.append(float(line['Validation loss']))
            f1_score.append(float(line['F1 Score']))
            agree_accuracy.append(int(line['Agree true']) / int(line['Agree total']))
            disagree_accuracy.append(int(line['Disagree true']) / int(line['Disagree total']))
            no_stance_accuracy.append(int(line['No stance true']) / int(line['No stance total']))
            not_relevant_accuracy.append(int(line['Not relevant true']) / int(line['Not relevant total']))
            true_sum = int(line['Agree true']) + int(line['Disagree true']) + int(line['No stance true']) + int(line['Not relevant true'])
            total_sum = int(line['Agree total']) + int(line['Disagree total']) + int(line['No stance total']) + int(line['Not relevant total'])
            total_accuracy.append(true_sum / total_sum)
        # draw now
        length = len(training_loss)
        x_ticks = np.arange(1, length+1, 1)
        plt.plot(list(x_ticks), training_loss, '.-')
        plt.plot(list(x_ticks), validation_loss, '.-')
        plt.plot(list(x_ticks), f1_score, '.-')
        plt.legend(['Training loss', 'Validation loss', 'F1 Score'], loc='upper left')
        plt.title(f"Train parameters: {titles[index]}")
        plt.xticks(x_ticks)
        plt.show()

        plt.plot(list(x_ticks), agree_accuracy, '.--')
        plt.plot(list(x_ticks), disagree_accuracy, '.--')
        plt.plot(list(x_ticks), no_stance_accuracy, '.--')
        plt.plot(list(x_ticks), not_relevant_accuracy, '.--')
        plt.plot(list(x_ticks), total_accuracy, '.-')
        for (i, accuracy) in enumerate(total_accuracy):
            plt.annotate(f"{round(accuracy, 2)}", xy=(i+1, accuracy), xytext=(-2, 5), textcoords='offset points')
        plt.legend(['Agree', 'Disagree', 'No stance', 'Not relevant', 'Total'], loc='lower right')
        plt.title(f"Accuracy: {titles[index]}")
        plt.xticks(x_ticks)
        plt.ylim(0, 1)
        plt.show()