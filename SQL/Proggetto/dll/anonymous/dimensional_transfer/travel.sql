create table travel
(
    travel_id    int auto_increment
        primary key,
    player_id    int                  null,
    dimension_id int                  null,
    permit       tinyint(1) default 0 null,
    constraint travel_ibfk_1
        foreign key (player_id) references player (player_id)
            on update cascade on delete cascade,
    constraint travel_ibfk_2
        foreign key (dimension_id) references dimension (dimension_id)
            on update cascade on delete cascade
);

create index dimension_id
    on travel (dimension_id);

create index player_id
    on travel (player_id);

