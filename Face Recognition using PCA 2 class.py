#Nmae: Tamal Majumder
#Entry No: 2023PHS7226

# Modules used
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import cv2


# -------------------------------------------------------------------------------------
# -------------------- Reading faces and making a vector out of it --------------------
# -------------------------------------------------------------------------------------

image_paths = list(paths.list_images(r"C:\Users\987ta\Desktop\2023PHS7226_TamalMajumder_A2\Data Two class\training"))

def processing_images(path, target_size=(100, 100)):  # Specify target size for resizing
    images = []
    for i in image_paths:
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, target_size)  # Resize image to target size
        images.append(img_resized.flatten())  # Flatten and append each resized image
    images_matrix = np.vstack(images)  # Stack flattened images vertically to form a matrix
    return images_matrix   

img_matrix = processing_images(image_paths) #This is the image matrix
#print("Original:",img_matrix)
img_matrix1=np.copy(img_matrix)

# -------------------------------------------------------------------------------------
"""
# Get dimensions of the image for checking if matrix being formed properly   ====== check for correctness
height, width = img_matrix.shape[:2] 
print("Image matrix Dimensions:")
print("Height:", height)
print("Width:", width)
"""
# -------------------------------------------------------------------------------------
# ------------------------------- Calculating mean image ------------------------------ 
# -------------------------------------------------------------------------------------

# Either use this ine or the for loop (This one is much much faster)

def mean_faces(image_matrix):
    return np.mean(image_matrix,axis=0)
#The Mean face matrix 
mean_face=mean_faces(img_matrix)

#The reshaped Mean face matrix for printing the image only
mean_face_reshape = np.reshape(mean_face, (100, 100))
#print(mean_face_reshape)
plt.imshow((mean_face_reshape), cmap='gray')
plt.title('Mean Face')
plt.axis('off')
plt.show()

# -------------------------------------------------------------------------------------
# --------------------- Finding centered image and covarience amtrix ------------------------ 
# -------------------------------------------------------------------------------------

#Now we calculate the average of all these face vectors and subtract it from each vector
#(Basically the centered image) -> How much every image differs (standard deviation)
img_matrix_centered=np.copy(img_matrix)
img_matrix = img_matrix - mean_face

# COVARIENCE Matrix: 
cov_mat=np.cov(img_matrix.T)

# --------------------------------------------------------------------------------------
# -------------------- Calculating The Eigenvalues and eigenvectors -------------------- 
# --------------------------------------------------------------------------------------

#Here the eigenfunctions give the projection

def calc_eigenfaces(covarience_matrix):
    return np.linalg.eigh(covarience_matrix)

Eig_val,Eig_f=calc_eigenfaces(cov_mat)
plt.plot(np.arange(len(Eig_val)),sorted(Eig_val,reverse=True))
#k=int(len(Eig_val)/2) #how many eigenvector to take
k=10
sorted_ind = np.argsort(Eig_val)[::-1]
top_k_ind=sorted_ind[:k]

Eig_val_sorted=Eig_val[top_k_ind]
Eig_f_sorted=Eig_f[:,top_k_ind]
s=25/21
#print(Eig_val)
plt.plot(np.arange(len(Eig_val)),sorted(Eig_val,reverse=True))
plt.xlabel(r'Number of eigenvalues $\longrightarrow$')
plt.ylabel(r'Eigenvalues $\longrightarrow$')
plt.xlim(0,12)
plt.show()
# So k=10 is a resonably good choice


# --------------------------------------------------------------------------------------
# ------------------------------------- Projection -------------------------------------
# --------------------------------------------------------------------------------------

image_paths = list(paths.list_images(r"C:\Users\987ta\Desktop\2023PHS7226_TamalMajumder_A2\Data Two class\training"))
num_image=s
def project_images(images, eigenfaces):
    get_img=processing_images(images)
    mean_face = mean_faces(get_img)
    centered_images = images - mean_face
    return np.dot(centered_images, eigenfaces)

projection=project_images(img_matrix1,Eig_f_sorted)
weights = np.dot(projection.T, img_matrix)
weights=weights/np.sum(weights)

def compute_weights(images, eigenfaces):
    projected_images = project_images(images, eigenfaces)
    return np.dot(projected_images, eigenfaces.T)
am=compute_weights(img_matrix1,Eig_f_sorted)

num_images=k
plt.figure(figsize=(100,100))

for i in range(k):
    plt.subplot(1, num_images, i + 1)
    eigf = np.reshape(weights[i], (100, 100))
    plt.imshow(eigf, cmap='gray')
    plt.axis('off')   
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------
# --------------------------- Weight and Accuracy calculation -------------------------- 
# --------------------------------------------------------------------------------------

# Weight Vector
# Function to process test images
def process_test_images(path, target_size=(100, 100)):
    images = []
    image_paths = list(paths.list_images(path))
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, target_size)
        images.append(img_resized.flatten())
    return np.array(images)

# Load test images and ground truth labels
test_images = process_test_images(r"C:\Users\987ta\Desktop\2023PHS7226_TamalMajumder_A2\Data Two class\test")

test_labels = np.arange(0,10,1)  
# Function to compute accuracy
def compute_accuracy(test_images, test_labels, weights, projection, mean_face, eigenfaces):
    num_correct = 0
    num_images = len(test_images)

    for i in range(num_images):
        # Preprocess the test image
        test_image = test_images[i]
        test_image_centered = test_image - mean_face
        test_projection = np.dot(test_image_centered, eigenfaces)

        # Computing the weights for the test image
        test_weights = np.dot(test_projection, eigenfaces.T)

        # Computing the Euclidean distance between the test weights and the training weights
        distances = np.linalg.norm(weights - test_weights, axis=1)

        # Finding the index of the minimum distance (nearest neighbor)
        min_index = np.argmin(distances)

        # Checking if the predicted label matches the ground truth label
        if min_index == test_labels[i]:
            num_correct += 1

    # Calculate accuracy
    accuracy = (num_correct / num_image) * 100.0
    return accuracy
# Accuracy
accuracy = compute_accuracy(test_images, test_labels, weights, projection, mean_face, Eig_f_sorted)
print("Accuracy:", accuracy, "%")
