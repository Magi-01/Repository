DROP DATABASE IF EXISTS testingDB;

CREATE DATABASE testingDB;
USE testingDB;

CREATE TABLE School(
    SchoolID INT PRIMARY KEY,
    SchoolName VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE Students(
    StudentsID INT PRIMARY KEY,
    StudentsName VARCHAR(255),
    StudentsSirName VARCHAR(255) NOT NULL
);

CREATE TABLE Professor(
    ProfessorID INT PRIMARY KEY,
    ProfessorName VARCHAR(255),
    ProfessorSirName VARCHAR(255) NOT NULL
);

CREATE TABLE Subjects(
    SubjectsID INT PRIMARY KEY,
    SubjectsName VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE Lessons(
    LessonsID INT PRIMARY KEY,
    LessonsName VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE Imatriculated(
    ImmatriculationID INT PRIMARY KEY,
    StudentsID INT NOT NULL,
    SchoolID INT NOT NULL,
    LessonsID INT,
    StatusInfo BOOLEAN DEFAULT FALSE,
    Foreign Key (StudentsID) REFERENCES Students(StudentsID) ON DELETE CASCADE,
    Foreign Key (SchoolID) REFERENCES School(SchoolID) ON DELETE CASCADE,
    Foreign Key (LessonsID) REFERENCES Lessons(LessonsID) ON DELETE CASCADE
);

CREATE TABLE Teaches(
    TeachingID INT PRIMARY KEY,
    ProfessorID INT,
    SubjectsID INT NOT NULL,
    Foreign Key (ProfessorID) REFERENCES Professor(ProfessorID) ON DELETE CASCADE,
    Foreign Key (SubjectsID) REFERENCES Subjects(SubjectsID) ON DELETE CASCADE
);

CREATE TABLE LessonsToSubject(
    ConnectionID INT PRIMARY KEY,
    LessonsID INT,
    SubjectsID INT NOT NULL,
    Foreign Key (LessonsID) REFERENCES Lessons(LessonsID) ON DELETE CASCADE,
    Foreign Key (SubjectsID) REFERENCES Subjects(SubjectsID) ON DELETE CASCADE
);

INSERT INTO School(SchoolID, SchoolName) VALUES 
(1, "Universita di Trieste"),
(2, "Universita di Udine"),
(3, "Universita di Padova");


INSERT INTO Students(StudentsID, StudentsName, StudentsSirName) VALUES 
(1, "Faddy", "Moha"),
(2, "Quanta", "Vally"),
(3, "Mixed", "Ula");

INSERT INTO Professor(ProfessorID, ProfessorName, ProfessorSirName) VALUES 
(1, "a", "aa"),
(2, "b", "bb"),
(3, "c", "cc");

INSERT INTO Subjects(SubjectsID, SubjectsName) VALUES 
(1, "Mate"),
(2, "Alge"),
(3, "Geol"),
(4, "Disegno"),
(5, "STORY"),
(6, "Poesia");

INSERT INTO Lessons(LessonsID, LessonsName) VALUES 
(1, "MAT"),
(2, "STORY"),
(3, "GEO");

INSERT INTO LessonsToSubject(ConnectionID, LessonsID, SubjectsID) VALUES
(1, 1, 1),
(2, 1, 2),
(3, 2, 3),
(4, 2, 4),
(5, 3, 5),
(6, 3, 6);

INSERT INTO Imatriculated(ImmatriculationID, StudentsID, SchoolID, LessonsID, StatusInfo) VALUES
(1, 1, 1, 1, TRUE),
(2, 2, 2, 2, TRUE),
(3, 3, 1, 3, TRUE);

INSERT INTO Teaches(TeachingID, ProfessorID, SubjectsID) VALUES
(1, 1, 1),
(2, 1, 2),
(3, 2, 3),
(4, 2, 4),
(5, 3, 5),
(6, 3, 6);

SELECT p.ProfessorID, p.ProfessorName as pName, p.ProfessorSirName as pSirName, s.SubjectsID, s.SubjectsName as sName
FROM Teaches t
JOIN professor p ON p.ProfessorID = t.ProfessorID
JOIN subjects s On s.SubjectsID = t.SubjectsID;

SELECT p.ProfessorID, p.ProfessorName as pName, p.ProfessorSirName as pSirName, s.SubjectsID, s.SubjectsName as sName
FROM Teaches t
RIGHT JOIN professor p ON p.ProfessorID = t.ProfessorID
JOIN subjects s On s.SubjectsID = t.SubjectsID;

SELECT p.ProfessorID, p.ProfessorName as pName, p.ProfessorSirName as pSirName, s.SubjectsID, s.SubjectsName as sName
FROM professor p
LEFT JOIN Teaches t ON p.ProfessorID = t.ProfessorID
LEFT JOIN subjects s On s.SubjectsID = t.SubjectsID
GROUP BY p.ProfessorID, p.ProfessorName, p.`ProfessorSirName`, s.SubjectsID, s.SubjectsName;

SELECT p.ProfessorSirName, COUNT(s.SubjectsID) AS SubjectsCount
FROM professor p
LEFT JOIN Teaches t ON p.ProfessorID = t.ProfessorID
LEFT JOIN subjects s ON s.SubjectsID = t.SubjectsID
GROUP BY p.ProfessorSirName
ORDER BY SubjectsCount DESC;
