show databases;

use classicmodels;

create procedure sp_countorders(customerNumber int)
begin
declare counting int;
select quantityOrdered into counting from orderdetails;
if counting = 0 then
    signal sqlstate '45000' set message_text = 'missing number';
end if;
end;

create procedure sp_emails(inout emaillist varchar(4000))
begin
    declare finished int default 0;
    declare v_email varchar(100) default '';

    declare exist_cursor cursor for
    select emails from customers;

    declare continue handler for not found set finished = 1;
    open exist_cursor;
    while(finished!=1) do
        fetch exist_cursor into v_emails;
        if finished = 0 then
            set emaillist = concat(v_email,';',emaillist)
        end if;
    end while;
    close exist_cursor;

end;