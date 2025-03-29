import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

#store csv file in a variable
file_path = "NFL.csv"
#create dataframe from csv
df=pd.read_csv(file_path, encoding='UTF-8')
#key_columns in original df
key_columns = ['Player', 'Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 'BMI']
#creating new dataframe with key columns
key_column_df=df[key_columns]
#statistics 
stat_key_columns=key_column_df.describe()
#test_measurable
test_measurable = ['Height', 'Weight','Sprint_40yd','Vertical_Jump','Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 'BMI']
#filling in nan with zeros
test_measure_df = df[test_measurable].fillna(0)

scaler = StandardScaler()
df_scaler = scaler.fit_transform(test_measure_df)
distance_measure_df= pd.DataFrame(df_scaler, columns=test_measurable)

similarity_matrix = cosine_similarity(df_scaler)
similarity_df = pd.DataFrame(similarity_matrix, index=df['Player'], columns=df['Player'])

query_players = ["Patrick Mahomes\MahoPa00", "Luke Kuechly\KuecLu00", "Chris Jones\JoneCh09"]

def top_similar_player(query, similarity_df, top_n=10):
    if query not in similarity_df.index:
        print(f"'{query}' Not Found")
        return 
    similar_players = similarity_df[query].sort_values(ascending =False).iloc[1:top_n+1]
    print(pd.DataFrame(similar_players).reset_index().rename(columns={"Player": "Player", query: "Similarity Score"}))
    
for query in query_players:
    print(f"\nTop 10 Most Similar Players to '{query}':")
    top_similar_player(query, similarity_df)
    







