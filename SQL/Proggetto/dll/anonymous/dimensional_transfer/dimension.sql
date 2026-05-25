create table dimension
(
    dimension_id   int auto_increment
        primary key,
    achievement_id int          null,
    name           varchar(255) not null,
    constraint dimension_ibfk_1
        foreign key (achievement_id) references achievement (achievement_id)
);

create index achievement_id
    on dimension (achievement_id);

