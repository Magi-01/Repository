show Databases;
Use mysql;

CREATE table Student(
    Matricola VARCHAR(11),
    Name varchar(45) NOT NULL,
    Sirname varchar(45) NOT NULL,
    PRIMARY KEY(Matricola), # primary key
    constraint notunique unique(Name,Sirname) #Name is tied to sirname, only one copy haviving each can exist
    );

#///////////////////////////////////////////////////

CREATE table Student1(
    Matricola VARCHAR(11) auto_increment, # *Creates an id column that auto update for each row added
    Name varchar(45) NOT NULL,
    Sirname varchar(45) NOT NULL,
    PRIMARY KEY(Matricola),
    constraint notunique unique(Name,Sirname) #Name is tied to sirname, only one copy haviving each can exist
    );
#* Auto-Increment = No recycle(if a student is delete, I will always have n+1), you can modify it and it will accept iff
#* the id doesn't exists in the column
#* Best used when one encounters a NULL in the PRIMARY KEY

#Check := a more complicated constraint. !You need to make sure if it exists

#///////////////////////////////////////////////////

CREATE table Student(
    Matricola VARCHAR(45) DEFAULT 'empty' #if the elements are not specified, != a row, then default value is 'empty'
    );

#///////////////////////////////////////////////////

CREATE table Student(
    Matricola VARCHAR(45) COMMENT 'blablabla'#comments on the column
    );

#///////////////////////////////////////////////////

CREATE table corsi(
    id VARCHAR(45) DEFAULT 'empty' PRIMARY KEY , #if the elements are not specified, != a row, then default value is 'empty'
    insegnante VARCHAR(45),
    voto int(4)
    );

#///////////////////////////////////////////////////

CREATE table Student(
    Matricola VARCHAR(11),
    Name varchar(45) NOT NULL,
    Sirname varchar(45) NOT NULL,
    corsi varchar(45) NOT NULL,
    PRIMARY KEY(Matricola), # primary key
    constraint notunique unique(Name,Sirname),
    FOREIGN KEY (corsi) references corsi(id), # Foreign Key constraining corsi in student to id of corsi
    FOREIGN KEY (name,Matricola) references corsi(insegnante)
    # Foreign Key constraining nome AND matricola, as one, in student to insegnante of corsi
    );

#///////////////////////////////////////////////////

CREATE table Student(
    Matricola VARCHAR(11),
    Name varchar(45) NOT NULL,
    Sirname varchar(45) NOT NULL,
    corsi varchar(45) NOT NULL,
    PRIMARY KEY(Matricola), # primary key
    constraint notunique unique(Name,Sirname),
    constraint FK_corsi FOREIGN KEY (Corsi) references corsi(id)
    # if you want to refernce the foreign key constraints(i.e turn on or off)
    );
SET FOREIGN_KEY_CHECKS = 0; # turn off the foreign key checks
SET FOREIGN_KEY_CHECKS = 1; # turn on the foreign key checks

#///////////////////////////////////////////////////

CREATE table Student(
    Matricola VARCHAR(11),
    Name varchar(45) NOT NULL,
    Sirname varchar(45) NOT NULL,
    corsi varchar(45) NOT NULL,
    PRIMARY KEY(Matricola), # primary key
    constraint notunique unique(Name,Sirname),
    constraint FK_corsi FOREIGN KEY (Corsi) references corsi(id)
    on delete set null on update cascade
    # If you want to refernce the foreign key constraints(i.e turn on or off)
    # When the corso is removed, cascade studenti and remove from corso everything on the id (in this case would refuse
    # as corsi is not NULL
    );

insert into Student(Matricola, Name, Sirname) VALUES ('SM3201434', 'Fadhla Mohamed','Mutua');

#///////////////////////////////////////////////////

alter table student
add participation varchar(45) AFTER Sirname;
# add 'name' 'type' after X
# add constraint 'name' foreign key ....
# drop froreign key 'name' (if the key has it)
# change 'currentname' 'newname' 'type'

#///////////////////////////////////////////////////

update student
set participation = 'LM320I'
where Matricola = 'SM3201434';

#///////////////////////////////////////////////////

select * from student;
# Print. Really selects all rows for use