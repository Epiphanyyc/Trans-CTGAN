# Load reference data
from ctgan.synthesizers.ctgan import CTGAN
from ctgan import load_demo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# third party
from sklearn.datasets import load_breast_cancer,load_diabetes

real_path = "../CTAB-GAN-main/Real_Datasets/creditcard2.csv"
real_data = pd.read_csv(real_path)

discrete_columns = []
for col in real_data.columns:
    if len(real_data[col].unique()) < 15:
        discrete_columns.append(col)
print(discrete_columns)
ctgan = CTGAN(epochs=150,num_layers=6,num_heads=8,batch_size=100,generator_lr=2e-4,discriminator_lr=2e-4)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
synthetic_data.to_csv('./fakeData_TransCTGAN-Credit2~.csv', index=False)
print(synthetic_data)


# correlation_matrix_real = real_data.corr()
# sns.heatmap(correlation_matrix_real, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix Heatmap')
# plt.show()


# correlation_matrix_fake = synthetic_data.corr()
# sns.heatmap(correlation_matrix_fake, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix Heatmap')
# plt.show()