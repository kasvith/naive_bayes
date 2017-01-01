import csv
import math
import random


def load_data(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)

    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]

    return dataset


def split_data(dataset, split_ratio):
    train_set_size = int(len(dataset) * split_ratio)
    train_set = []
    original = list(dataset)

    while len(train_set) < train_set_size:
        index = random.randrange(0, len(original))
        train_set.append(original.pop(index))

    return [train_set, original]


def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vec = dataset[i]
        if vec[-1] not in separated:
            separated[vec[-1]] = []
        separated[vec[-1]].append(vec)

    return separated


def mean(data):
    return sum(data) / float(len(data))


def std_dev(data):
    avg = mean(data)
    variance = sum([pow(x - avg, 2) for x in data]) / float(len(data) - 1)
    return math.sqrt(variance)


def summerize(data):
    summerized = [(mean(attr), std_dev(attr)) for attr in zip(*data)]
    del summerized[-1]
    return summerized


def summerize_by_class(dataset):
    seperated = separate_by_class(dataset)
    summerized = {}

    for className, instance in seperated.items():
        summerized[className] = summerize(instance)

    return summerized


def calculate_probability(x, _mean, _stddev):
    expo = math.exp(-1 * math.pow(x - _mean, 2) / (2 * math.pow(_stddev, 2)))
    return (1 / math.sqrt(2 * math.pow(_stddev, 2) * math.pi)) * expo


def calculate_class_probabilities(summeries, input_vector):
    probabilities = {}
    for class_value, class_summeries in summeries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summeries)):
            _mean, _stddev = class_summeries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, _mean, _stddev)
    return probabilities


def predict(summeries, input_vector):
    probabilities = calculate_class_probabilities(summeries, input_vector)
    best_label, best_probability = None, -1
    for class_val, probability in probabilities.items():
        if best_label is None or probability > best_probability:
            best_label = class_val
            best_probability = probability
    return best_label


def get_predictions(summeries, test_set):
    predictions = []
    for i in range(len(test_set)):
        predictions.append(predict(summeries, test_set[i]))
    return predictions


def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1

    return correct*100/float(len(test_set))


def main():
    print("Gaussian Naive Bayes classifier\n")

    filename = "pima-indians-diabetes.data.csv"
    dataset = load_data(filename)
    print("Loaded {0} with {1} rows".format(filename, len(dataset)))
    split_ratio = 0.67

    train_set, testing_set = split_data(dataset, split_ratio)
    print("{0} rows were separated into {1} training set and {2} testing set"
          .format(len(dataset), len(train_set), len(testing_set)))

    sep = separate_by_class(train_set)
    print("{0} classes were separated".format(len(sep.keys())))

    for cls, rows in sep.items():
        print("\t{0} class has {1} elements".format(cls, len(rows)))

    summerized = summerize_by_class(train_set)
    pred = get_predictions(summerized, testing_set)
    print("Accuracy {0}%".format(get_accuracy(testing_set, pred)))


if __name__ == '__main__':
    main()
