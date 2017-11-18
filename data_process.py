def process_train_data(data):
  labels = data['label']
  features = data.drop('label', axis=1)

  return features[0:10000], labels[0:10000], features[10000:42000], labels[10000:42000]
