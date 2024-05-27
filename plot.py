import matplotlib.pyplot as plt


def plot_image(train,target):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(train)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(target)
    plt.title("Enhanced Image")
    plt.axis('off')

    plt.show()
