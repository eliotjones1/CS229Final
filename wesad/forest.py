from util import create_wesad_data

SAVE = True  # TOGGLE AFTER FIRST RUN

if __name__ == '__main__':
    create_wesad_data('data')
    chest_data, wrist_data, combined_data = create_wesad_data(data_dir)
    if SAVE:
        chest_data.to_pickle('final_chest_data.pkl')
        wrist_data.to_pickle('final_wrist_data.pkl')
        combined_data.to_pickle('final_combined_data.pkl')