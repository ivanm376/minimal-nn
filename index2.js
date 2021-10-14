// i     w     r
// 0.6   1   > 1
// 0.6   0.5 > 0.6
// 0.6   0   > 0

let input = 0.6;
let w1 = 0.6;
let result;
// w > x, r > y
const calc = (x1, y1, x2, y2, x3, y3) => {
  const denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
  const a = x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2);
  const b = Math.pow(Math.pow(Math.pow(x3, 2 * (y1 - y2) + x2), 2 * (y3 - y1) + x1), 2 * (y2 - y3));
  const c = x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3;
  return [a, b, c];
};

const projectWeight = (input, weight) => {
  if (weight >= 0.5) {
    // weight === 1    > multiplier === 1
    // weight === 0.75 > multiplier === 0.5
    // weight === 0.5  > multiplier === 0
    return input + (1 - input) * (weight * 2 - 1);
  } else {
    // weight === 0.499 > multiplier === 0.9999 ~
    // weight === 0.25  > multiplier === 0.5
    // weight === 0     > multiplier === 0
    return input * weight * 2;
  }
};

//(a+b)*(c+d) === a*c + b*c + a*d + b*d
const t = (a, b, c, d) => a * c + a * d + b * c + b * d;
const t2 = (a, b, c, d) => (a + b) * (c + d);

const canvas = document.createElement('canvas');
document.body.appendChild(canvas);
const ctx = canvas.getContext('2d');
ctx.fillStyle = 'black';
const draw = input => {
  for (let x = 0; x <= 100; x++) {
    const y = projectWeight(input, x / 100);
    ctx.fillRect(x, Math.floor(y * 100), 1, 1); // dot
  }
};
for (let i = 0; i <= 10; i++) {
  draw(i / 10);
}

// ctx.fillStyle = 'blue';
// const input = 0.3;
// const resultHalf = projectWeight(input, 0.5);
// const testWeight = 0.7;
// const [a, b, c] = calc(0, 0, testWeight, resultHalf, 1, 1);
// for (let x = 0; x <= 100; x += 1) {
//   const y = a * Math.pow(x / 100, 2) + b * (x / 100) + c;
//   ctx.fillRect(x, y * 100, 1, 1); // dot
// }

const adjustWeight_old = (input, oldWeight, result, expected) => {
  const error = round((expected - result) * learnRate);
  const newExpectedResult = round(result + error);

  // const weightsDelta = round(error * result * (1 - result)); // youtu.be/HA-F6cZPvrg?t=14m30s
  // const newWeight = round(oldWeight - input * weightsDelta * learnRate);
  const oldProjectedResult = round(projectWeight(input, oldWeight));
  const newProjectedResult = round((newExpectedResult / result) * oldProjectedResult);
  const calculatedWeight = round(calculateWeight(input, newProjectedResult));
  console.log(
    `${expected} :\t${error}\t${result}\t-> ${newExpectedResult}\t${oldProjectedResult}\t-> ${newProjectedResult}\tinput:${input}\tw:${calculatedWeight}`
  );
  return calculatedWeight;
  if (result > expected) {
    // console.log(weightsDelta, `${input}\t${result}\t${expected}\t${oldWeight}\t-> ${newWeight}`);
  }
  // console.log(error, `old:${oldWeight} new:${newWeight}`, weightsDelta);
  // return round(newWeight);

  // TODO reduce remaining part of weight eg 0.6 + (-1 * 0.4*learnRate)
};

const projectWeight_old2 = (input, weight) => (input * (weight + 0.5) * 2) / 3;
const calculateWeight_old2 = (input, result) => (result * 3) / input / 2 - 0.5;
const calculateInput_old2 = (result, weight) => (result * 3) / 2 / (weight + 0.5);

const run_old = (train, weights, input, expected = []) => {
  const [i1, i2] = input;
  const [w1, w2, w3, w4, w5, w6, w7, w8] = weights;
  // const n1 = sigmoid(i1 * w1 + i2 * w3); // neurons
  // const n2 = sigmoid(i1 * w2 + i2 * w4);
  // const r1 = sigmoid(n1 * w5 + n2 * w7); // result
  // const r2 = sigmoid(n1 * w6 + n2 * w8);

  const n1 = round((projectWeight(i1, w1) + projectWeight(i2, w3)) / 2);
  const n2 = round((projectWeight(i1, w2) + projectWeight(i2, w4)) / 2);
  const r1 = round((projectWeight(n1, w5) + projectWeight(n2, w7)) / 2);
  const r2 = round((projectWeight(n1, w6) + projectWeight(n2, w8)) / 2);

  let logString = `input:[${i1},${i2}] result:[${r1},${r2}]`;
  let logWeights = `weights:\t${weights.join('\t')}`;
  if (train) {
    const updatedWeights = weights.slice();
    const [e1, e2] = expected;
    logString += ` expected:[${e1},${e2}]`;
    // updatedWeights[4] = adjustWeight(n1, w5, r1, e1); // updatedWeights[5] = adjustWeight(n1, w6, r2, e2);
    // updatedWeights[6] = adjustWeight(n2, w7, r1, e1); // updatedWeights[7] = adjustWeight(n2, w8, r2, e2);

    updatedWeights[4] = adjustWeight(w5, r1 < e1);
    updatedWeights[5] = adjustWeight(w6, r2 < e2);
    updatedWeights[6] = adjustWeight(w7, r1 < e1);
    updatedWeights[7] = adjustWeight(w8, r2 < e2);

    // const newN1 = sigmoid(r1 * w5 + r2 * w6); // ? use e1/e2 instead of r1/r2
    // const newN2 = sigmoid(r1 * w7 + r2 * w8);

    const newN1 = round((calculateInput(e1, updatedWeights[4]) + calculateInput(e2, updatedWeights[5])) / 2);
    // console.log(e1, updatedWeights[4], e2, updatedWeights[5], ':', n1, newN1);
    const newN2 = round((calculateInput(e1, updatedWeights[6]) + calculateInput(e2, updatedWeights[7])) / 2);

    updatedWeights[0] = adjustWeight(w1, n1 < newN1);
    updatedWeights[1] = adjustWeight(w2, n2 < newN2);
    updatedWeights[2] = adjustWeight(w3, n1 < newN1);
    updatedWeights[3] = adjustWeight(w4, n2 < newN2);

    // const newN1 = sigmoid(r1 * w5 + r2 * w6); // ? use e1/e2 instead of r1/r2
    // const newN2 = sigmoid(r1 * w7 + r2 * w8);
    // updatedWeights[0] = adjustWeight(n1, newN1, w1, i1); // updatedWeights[1] = adjustWeight(n2, newN2, w2, i1);
    // updatedWeights[2] = adjustWeight(n1, newN1, w3, i2); // updatedWeights[3] = adjustWeight(n2, newN2, w4, i2);
    logWeights += `\nupdatedWeights:\t${updatedWeights.join('\t')}`;
    // setTimeout(() => run(0, updatedWeights, input), 100);
    return updatedWeights;
  }
  console.log(`${logString}\n${logWeights}`);
};
