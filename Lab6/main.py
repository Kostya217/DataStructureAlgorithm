import math
from tqdm import tqdm


def logical_and(x1: int, x2: int) -> int:
    w1, w2 = 1, 1
    t = 1.5
    s = x1 * w1 + x2 * w2
    if s >= t:
        return 1
    else:
        return 0


def logical_or(x1: int, x2: int) -> int:
    w1, w2 = 1, 1
    t = 0.5
    s = x1 * w1 + x2 * w2
    if s >= t:
        return 1
    else:
        return 0


def logical_not(x: int) -> int:
    w = -1.5
    t = -1
    s = x * w
    if s >= t:
        return 1
    else:
        return 0


def xor(x1: int, x2: int) -> int:
    w11, w12 = 1, -1
    w21, w22 = -1, 1
    w31, w32 = 1, 1
    t = -0.5

    if x1 * w11 + x2 * w12 < t:
        y1 = 0
    else:
        y1 = 1

    if x1 * w21 + x2 * w22 < t:
        y2 = 0
    else:
        y2 = 1

    if y1 * w31 + y2 * w32 < t:
        return 0
    else:
        return 1


# time series forecasting
class TimeSeriesForecasting:
    def __init__(self):
        self.weights = [1 for _ in range(3)]
        self.inputs = []
        self.correct_y = []
        self.error = None
        self.learning_rate = 0.85  # 0.8, 0.35, 0.2
        self.n = 0

    # Set incoming signal and correct value
    def set_data(self, incoming_signal: list, value: list) -> None:
        self.inputs = incoming_signal.copy()
        self.correct_y = value.copy()
        self.n = len(value)

    # Calculation of the weighted sum of the i-th set of input data
    def get_s(self, x):
        return sum([x[i] * self.weights[i] for i in range(len(x))])
    # x1 * w1 + x2 * w2...

    # Calculation of the predicted value of the i-th member of the time series xi;
    def sigmoid(self, s):
        return 1 / (1 + math.exp(-s)) * 10  # 10, 24, 5

    # Calculation of total error (sum of squared error)
    def mean_square_error(self, y_predicts):
        return sum([(y_predicts[i] - self.correct_y[i]) ** 2 for i in range(self.n)])

    # Calculation of the derivative error value
    def derivative_error(self, y_predict, y_true, s, xi):
        return (y_predict - y_true) * (math.exp(-s) / (1 + math.exp(-s)) ** 2) * xi

    # Calculation of the corrective weighting factor
    def correction_w(self, derivative_error_i):
        return -self.learning_rate * derivative_error_i

    # Calculation of the average value of delta w
    def average_delta_w(self, delta_wi: list):
        return sum(delta_wi) / self.n

    def training(self):
        s = []
        y_predicts = []
        de1 = []
        de2 = []
        de3 = []
        dw1 = []
        dw2 = []
        dw3 = []
        msq0 = 0
        pbar = tqdm(total=1_000_000, ncols=100, desc="Learning")
        for i in range(0, 1_000_000):
            for j in range(self.n):
                s.append(self.get_s(self.inputs[j]))
                y_predicts.append(self.sigmoid(s[j]))

            msq = self.mean_square_error(y_predicts)

            if abs(msq - msq0) < 0.0001:
                print("Training is over!")
                pbar.update(1_000_000 - i)
                break

            for j in range(self.n):
                de1.append(self.derivative_error(
                    y_predict=y_predicts[j],
                    y_true=self.correct_y[j],
                    s=s[j],
                    xi=self.inputs[j][0]
                ))
                de2.append(self.derivative_error(
                    y_predict=y_predicts[j],
                    y_true=self.correct_y[j],
                    s=s[j],
                    xi=self.inputs[j][1]
                ))
                de3.append(self.derivative_error(
                    y_predict=y_predicts[j],
                    y_true=self.correct_y[j],
                    s=s[j],
                    xi=self.inputs[j][2]
                ))

                dw1.append(self.correction_w(derivative_error_i=de1[j]))
                dw2.append(self.correction_w(derivative_error_i=de2[j]))
                dw3.append(self.correction_w(derivative_error_i=de3[j]))
                # print(dw1)

            self.weights[0] += sum(dw1) / self.n
            self.weights[1] += sum(dw2) / self.n
            self.weights[2] += sum(dw3) / self.n

            msq0 = msq

            s.clear()
            y_predicts.clear()
            de1.clear()
            de2.clear()
            de3.clear()
            dw1.clear()
            dw2.clear()
            dw3.clear()
            pbar.update(1)

    def testing(self):
        y_predicts = []
        for j in range(self.n):
            y_predicts.append(self.sigmoid(s=self.get_s(self.inputs[j])))
        return y_predicts


def additional_task(x1: int, x2: int, x3: int) -> int:
    w1, w2, w3 = -1, 1, 1
    s = x1 * w1 + x2 * w2 + x3 * w3

    if s > -1:
        return 1
    else:
        return 0


def main():
    print(f"Logical AND, x1 = 0, x2 = 1: {logical_and(0, 1)}")
    print(f"Logical OR, x1 = 0, x2 = 0: {logical_or(0, 0)}")
    print(f"Logical NOT, x = 1: {logical_not(1)}")
    print(f"XOR, x1 = 1, x2 = 1:  {xor(1, 1)}")
    print(f"Additional task, x1 = 0, x2 = 1, x3 = 0: {additional_task(1, 1, 1)}")

    print(f"Time series forecasting:")
    tsf = TimeSeriesForecasting()
    tsf.set_data(
        incoming_signal=[
            [0.13, 5.97, 0.57],
            [5.97, 0.57, 4.02],
            [0.57, 4.02, 0.31],
            [4.02, 0.31, 5.55],
            [0.31, 5.55, 0.15],
            [5.55, 0.15, 4.54],
            [0.15, 4.54, 0.65],
            [4.54, 0.65, 4.34],
            [0.65, 4.34, 1.54],
            [4.34, 1.54, 4.70]
        ],
        value=[4.02, 0.31, 5.55, 0.15, 4.54, 0.65, 4.34, 1.54, 4.70, 0.58]
    )
    tsf.training()

    tsf.set_data(
        incoming_signal=[
            [1.54, 4.70, 0.58],
            [4.70, 0.58, 5.83]
        ],
        value=[5.83, 0.03]
    )
    print(f"incoming signal: {tsf.inputs}")
    print(f"Correct value: {tsf.correct_y}")
    print(f"predicting value: {tsf.testing()}")
    # [5.83, 0.03]


if __name__ == '__main__':
    main()
