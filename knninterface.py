import sys
import pickle
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit

# Load the pickled model and data
with open('knn_model.pkl', 'rb') as f:
    knn, vectorizer, unique_sequences = pickle.load(f)

def find_nearest_neighbors(prospect_id, unique_sequences, knn_model, vectorizer):
    # Check if the ProspectID exists in the dataset
    if prospect_id not in unique_sequences['ProspectID'].values:
        return None
    
    # Get the event sequence for the given ProspectID
    event_sequence = unique_sequences.loc[unique_sequences['ProspectID'] == prospect_id, 'EventSequence'].values[0]
    
    # Vectorize the event sequence
    event_vector = vectorizer.transform([event_sequence])
    
    # Find the nearest neighbors
    distances, indices = knn_model.kneighbors(event_vector)
    
    # Get the nearest neighbor ProspectIDs and their event sequences
    neighbors = unique_sequences.iloc[indices[0]]
    
    return neighbors

class NeighborFinderApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel('Enter Prospect ID:')
        layout.addWidget(self.label)

        self.input_field = QLineEdit(self)
        layout.addWidget(self.input_field)

        self.button = QPushButton('Find Neighbors', self)
        self.button.clicked.connect(self.show_neighbors)
        layout.addWidget(self.button)

        self.result_area = QTextEdit(self)
        self.result_area.setReadOnly(True)
        layout.addWidget(self.result_area)

        self.setLayout(layout)
        self.setWindowTitle('Neighbor Finder')
        self.setGeometry(100, 100, 600, 400)

    def show_neighbors(self):
        prospect_id = self.input_field.text()
        neighbors = find_nearest_neighbors(prospect_id, unique_sequences, knn, vectorizer)
        
        if neighbors is not None:
            result_text = f"Neighbors for ProspectID {prospect_id}:\n\n"
            for index, row in neighbors.iterrows():
                result_text += f"ProspectID: {row['ProspectID']}\nStatus: {row['Status']}\nEventSequence: {row['EventSequence']}\n\n"
            self.result_area.setText(result_text)
        else:
            self.result_area.setText(f"ProspectID {prospect_id} not found in the dataset.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NeighborFinderApp()
    ex.show()
    sys.exit(app.exec_())
