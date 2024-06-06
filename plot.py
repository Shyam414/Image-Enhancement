import matplotlib.pyplot as plt

def plot_image_3(train,target,Final):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(train)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(target)
    plt.title("Enhanced Image")
    plt.axis('off')

    
    plt.subplot(1, 3, 3)
    plt.imshow(Final)
    plt.title("As to be")
    plt.axis('off')


    plt.show()


def plot_image_2(train,target):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(train)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(target)
    plt.title("Target Image")
    plt.axis('off')

    plt.show()
