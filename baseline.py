from collections import defaultdict

def readingFile(filename):
    f = open(filename, "r")
    data = []
    for row in f:
        if row.startswith("userId"):
            continue
        r = row.split(',')
        e = [int(r[0]), int(r[1]), float(r[2])]
        data.append(e)
    return data


train = readingFile("rating_train.csv")

allRatings = []
userRatings = defaultdict(list)
for l in train:
    allRatings.append(l[2])
globalAverage = sum(allRatings) / len(allRatings)
print("alpha: "+str(globalAverage))


test = readingFile("rating_test.csv")

def calc_mse(test):
    predictions = [globalAverage]*30000
    test_val = [l[2] for l in test]
    def squaredDiff(x, y):
        sum = 0.0
        for a,b in zip(x,y):
            sum += 1.0*(a-b)*(a-b)
        return sum
    mse = squaredDiff(predictions, test_val) / len(test_val)
    print("baseline mse: "+str(mse))

calc_mse(test)