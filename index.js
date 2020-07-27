require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const knn = (features, label, predictionPoint, k) => {
  return (
    features
      .sub(predictionPoint)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => {
        return a[0] > b[0] ? 1 : -1;
      })
      .slice(0, k)
      .reduce((acc, pair) => {
        return acc + pair[1];
      }, 0) / k
  );
};
let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long"],
    labelColumns: ["price"],
  }
);
features = tf.tensor(features);
labels = tf.tensor(labels);
//testFeatures = tf.tensor(testFeatures);
//testLabels = tf.tensor(testLabels);
const result = knn(features, labels, tf.tensor(testFeatures[0]), 10);
console.log(result);
// console.log(testFeatures);
// console.log(testLabels);
// const k = 2;
// const features = tf.tensor(
//   [-121, 47],
//   [-121.2, 46.5],
//   [-122, 46.4],
//   [-120.9, 46.7]
// );
// const labels = tf.tensor([[200], [250], [215], [240]]);
// const predictionPoint = tf.tensor([-121, 47]);
//
// features
//   .sub(predictionPoint)
//   .pow(2)
//   .sum(1)
//   .pow(0.5)
//   .expandDims(1)
//   .concat(labels, 1)
//   .unstack()
//   .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
//   .slice(0, k)
//   .reduce((acc, pair) => {
//     return acc + pair.get(1);
//   }, 0) / k;
