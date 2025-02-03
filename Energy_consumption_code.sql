

SELECT * FROM renewableenergy.energy_consumption
limit 10;


describe renewableenergy.energy_consumption;

-- Step 1: Add new columns with the correct data types for Date and Time
ALTER TABLE energy_consumption
ADD COLUMN NewDate DATE,
ADD COLUMN NewTime TIME;

-- Step 2: Convert the existing text-based Date and Time to correct formats
UPDATE energy_consumption
SET NewDate = STR_TO_DATE(Date, '%W, %M %e, %Y'),
    NewTime = STR_TO_DATE(Time, '%h:%i:%s %p');

-- Step 3: Remove old Date and Time columns and rename NewDate & NewTime
ALTER TABLE energy_consumption
DROP COLUMN Date,
DROP COLUMN Time,
CHANGE COLUMN NewDate Date DATE,
CHANGE COLUMN NewTime Time TIME;

SHOW COLUMNS FROM energy_consumption;

-- Step 4: Modify table structure to match the new format

ALTER TABLE energy_consumption
ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY,
MODIFY COLUMN ConsumerType varchar(50),
MODIFY COLUMN EnergyConsumed_kWh FLOAT;




