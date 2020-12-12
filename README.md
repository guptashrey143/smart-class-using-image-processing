# smart-class-using-image-processing
The files consists of 4 terminal prgrams, one Haar-Cascade face classifier and 3 databases which use to keep the data of total students in an institution, students present in the class and class chart for a specific classroom.

Registration.pynb : It is one of the terminal programs which we use to edd a students detail to the database and capture the students images to train a classifier for face recognition.

Classroom.pynb : It is the second terminal programs which we use to identify the student entering the face on the basis of face recogntion using the classifier which we trained earlier. It also add the name of the student entering the classroom and store the details in a csv file.

Teacher Podium.csv : It is third terminal program on which the names of the students currently attending the class would be displayed and after the class is over the teacher could clear the data as required.

Batch Scheduling.pynb : It is the fourth terminal program and is used to add a class to a csv file.

students_data.csv : It is the overall data of all the students in the institution by their name, enrollment id, and batch.

time chart.csv : It keeps the time chart of the classes that are scheduled in that specific room.

present_students.csv : It keeps the names and enrollment ids of the students of the students present in the class.
