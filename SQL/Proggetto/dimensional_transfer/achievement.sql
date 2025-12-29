create table achievement
(
    achievement_id int auto_increment
        primary key,
    name           varchar(255)         not null,
    state          tinyint(1) default 0 null
);

