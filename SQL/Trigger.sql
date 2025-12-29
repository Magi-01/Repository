CREATE TABLE credit_change (
id int(11) NOT NULL AUTO_INCREMENT,
customer_number int(11) NOT NULL,
lastname varchar(50) NOT NULL,
changed_on datetime DEFAULT NULL,
changedBy varchar(50) DEFAULT NULL,
credit_limit int(11) Not null,
action varchar(50) DEFAULT NULL,
PRIMARY KEY (id)
);

delimiter $$

create trigger change_credit_limit
after update on credit_change
for each row
begin
    if new.credit_limit > old.credit_limit then
        signal sqlstate '45001' set message_text = 'increment found';
    end if;

end $$
delimiter ;