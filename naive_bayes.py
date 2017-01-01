from histogram import Histogram

testing_set = []
stats = {}
equal_probability_for_class = 0


def calculate_probability(words):
    probabilities = {}
    for key in stats.keys():
        probabilities[key] = 1 * equal_probability_for_class

    for cls, words_list in stats.items():
        prob = probabilities[cls]
        for word in words:
            if word in words_list.keys():
                prob *= words_list[word]
        probabilities[cls] = prob

    return probabilities


def predict_class(probabilities):
    max_prob = 0
    pred_cls = None
    for cls, prob in probabilities.items():
        if max_prob < prob:
            max_prob = prob
            pred_cls = cls

    return pred_cls


def get_accuracy():
    passed = 0
    num_tot = float(len(testing_set))

    for line in testing_set:
        data = line.split(":")
        cls = data[0]
        words = (data[1].split())
        words = Histogram.remove_unwanted(words)
        words = Histogram.switch_simple_case(words)
        probs = calculate_probability(words)

        if cls == predict_class(probs):
            passed += 1

    return float(passed * 100) / num_tot


if __name__ == '__main__':
    h = Histogram(split_ratio=0.89)
    testing_set = h.get_test_set()
    stats = h.get_statisics_data()
    equal_probability_for_class = 1 / h.get_num_classes()
    accuracy = get_accuracy()

    # display results
    print("Naive Bayes Classifier\n")
    print("Loaded {0} lines from {1}".format(len(testing_set) + len(h.get_train_set()), "dialog.txt"))
    print("Separated into {0} of training set and {1} of testing set".format(len(h.get_train_set()), len(testing_set)))
    print("Average accuracy of classifier is {0}%".format(get_accuracy()))
