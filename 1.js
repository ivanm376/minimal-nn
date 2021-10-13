// const error = 0.05
const learnRate = 0.1;
const [w1, w2, w3, w4, w5, w6, w7, w8] = [0.42, 0.17, 0.78, 0.55, 0.22, 0.99, 0.12, 0.88]; // random weights
const round = x => Math.round(x * 100) / 100; // 0.42851.. >> 0.43
const sigmoid = x => round(1 / (1 + Math.exp(-x))); // range: 0 - 1

const i1 = 0.2;
const i2 = 0.5; // input values
const n1 = sigmoid(i1 * w1 + i2 * w2); // neurons
const n2 = sigmoid(i1 * w3 + i2 * w4);
const o1 = sigmoid(n1 * w5 + n2 * w6); // output
const o2 = sigmoid(n1 * w7 + n2 * w8);

console.log(n1, n2, o1, o2);
//   0.54  0.61  0.67  0.65

const adjustWeight = (actual, expected, oldWeight, output) => {
  const error = Math.abs(actual - expected);
  const weightsDelta = round(error * actual * (1 - actual)); // https://youtu.be/HA-F6cZPvrg?t=14m30s
  const new_weight = round(oldWeight - output * weightsDelta * learnRate);
  console.log(error, weightsDelta, `new_weight ${new_weight}`);
};

const e1 = 0;
const e2 = 1; // expected results

adjustWeight(0.69, 0, 0.5, 0.77);

const propagated_error = weight * weightsDelta;

adjustWeight(o1, e1, w5, n1);
adjustWeight(o1, e1, w6, n2);
adjustWeight(o2, e2, w7, n1);
adjustWeight(o2, e2, w8, n2);
