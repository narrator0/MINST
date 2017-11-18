def process_train_data(data):
  labels = data['label']
  features = data.drop('label', axis=1)

  return features[0:37800], labels[0:37800], features[37801:42000], labels[37801:42000]
