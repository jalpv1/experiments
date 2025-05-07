from torch.utils.data import DataLoader, WeightedRandomSampler
import  numpy as np
import cv2
from skimage import feature

class_names = ['Abstract_Expressionism','Baroque','Cubism', 'Fauvism', 'Impressionism','Minimalism','Naive_Art_Primitivism','Pointillism','Rococo','Ukiyo_e']
num_classes = 10

def extract_labels(train_loader):
  labels_ = []
  for inputs, labels in train_loader:
      labels_.append(labels.numpy())
  return labels_

def data_load(data):
    dataset = int(len(data))
    dataset_loader = DataLoader(data, shuffle=True)
    y = extract_labels(dataset_loader)
    y = np.concatenate(y)
    unique, counts = np.unique(y, return_counts=True)

    class_weights = [1.0 / c for c in counts]
    sample_weights = [class_weights[i] for i in y]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    #train_loader = DataLoader(data)
    train_loader = DataLoader(data, sampler=sampler)
    print("with w")
    return train_loader


def rgb_to_cielab(image):
    #print(np.shape(image))
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

def rgb_to_cieYCrCb(image):
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

def rgb_to_hsv(image):
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def extract_color_features(image):
    features = []
    for color_space_func in [rgb_to_cielab, rgb_to_cieYCrCb, rgb_to_hsv]:
        transformed_image = color_space_func(image)
        mean = np.mean(transformed_image, axis=(0, 1))
        std = np.std(transformed_image, axis=(0, 1))
        features.extend(mean)
        features.extend(std)
    return features

def extract_lbp(image):
   image = np.transpose(image, (1, 2, 0))
   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   sample_detailed = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
   sample_broad = feature.local_binary_pattern(image, P=24, R=3, method='uniform')
   hist_detailed = np.histogram(sample_detailed, bins=10)[0]
   hist_broad = np.histogram(sample_broad, bins=26)[0]
   return  np.concatenate([hist_detailed, hist_broad])

def extract_color_palette(image, num_colors=5):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    pixels = image_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None,
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                    attempts=10,
                                    flags=cv2.KMEANS_RANDOM_CENTERS)
   # print(centers)
    centroids = centers.flatten().tolist()

    return  centroids
device = "cuda"
def extract_features(model, train_loader):
  features = []
  labels_ = []
  map_img_feature = {}
  for inputs, labels in train_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      feature = model(inputs)
      feature = feature.mean((2,3)).squeeze()
      cnn_feature = feature.detach().cpu().numpy()
      print(np.shape(cnn_feature))
      color_feature = extract_color_features(inputs.squeeze().detach().cpu().numpy())
      color_palette = extract_color_palette(inputs.squeeze().detach().cpu().numpy(), num_colors=15)
      lbp_features = extract_lbp(inputs.squeeze().detach().cpu().numpy())
      concatenated_vector = np.concatenate((cnn_feature, color_feature, color_palette,lbp_features))
      print(np.shape(concatenated_vector))
      features.append(concatenated_vector)
      labels_.append(labels.cpu().numpy())
      print("----------------------------------")

  return features, labels_, map_img_feature


def save_pickle(train_data,train_labels,name ):
    import pickle
    with open(f'X_{name}.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'Y_{name}.pickle', 'wb') as handle:
        pickle.dump(train_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved")

def load_pickle():
    import pickle
    with open('traindata.pickle', 'rb') as handle:
        train_data = pickle.load(handle)
    with open('train_labels.pickle', 'rb') as handle:
        train_labels = pickle.load(handle)
    return train_data, train_labels
