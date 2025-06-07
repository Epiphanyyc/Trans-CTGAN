from ctgan.synthesizers.ctgan import CTGAN,EnhancedCTGAN
from ctgan import load_demo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#real_path = "../CTAB-GAN-main/Real_Datasets/Adult3.csv"
#real_path = "../CTAB-GAN-main/Real_Datasets/Credit.csv"
real_path = '../CTGAN-main/CTGAN-main/examples/csv/train_clean.csv'
#real_path = "../synthcity-main/tutorials/covertype_preprocessed.csv"
data = pd.read_csv(real_path)

discrete_columns = [
    'Gender',
    'Ethnicity',
    'ParentalEducation',
    'Tutoring',
    'ParentalSupport'
]


ctgan = EnhancedCTGAN(epochs=300)
ctgan.fit(data)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
print(synthetic_data)
synthetic_data.to_csv('./fakeData_trans-CTGAN--student.csv', index=False)

correlation_matrix_real = data.corr()
sns.heatmap(correlation_matrix_real, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()


correlation_matrix_fake = synthetic_data.corr()
sns.heatmap(correlation_matrix_fake, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()