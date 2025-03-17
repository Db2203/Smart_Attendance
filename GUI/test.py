import pickle

# Load the student encodings
with open("student_encodings.pkl", "rb") as f:
    student_encodings = pickle.load(f)

# Print the keys (Reg Nos) and the number of encodings for each student
for reg_no, encodings in student_encodings.items():
    print(f"Reg No: {reg_no}, Encodings Count: {len(encodings)}")
