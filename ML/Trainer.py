from Generate_synthetic_data import generate_synthetic_data
from Best_model_selection_trainer import train_and_save_model

synthetic_df = generate_synthetic_data(n_samples=500)
synthetic_df.to_csv('ML/Training_data/synthetic_data.csv', index=False)

train_and_save_model(synthetic_df, model_path='social_support_model_best')
