create table player_item
(
    player_item_id int auto_increment
        primary key,
    player_id      int                  null,
    item_id        int                  null,
    legality       tinyint(1) default 1 null,
    constraint player_item_ibfk_1
        foreign key (player_id) references player (player_id)
);

create index player_id
    on player_item (player_id);

