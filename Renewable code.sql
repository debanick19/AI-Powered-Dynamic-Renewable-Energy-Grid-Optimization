
create database RenewableEnergy;
use  RenewableEnergy;



SELECT * FROM renewableenergy.renewable_energy;

-- Step 1: Add a new column 'NewDate' with DATE type
ALTER TABLE renewable_energy
ADD COLUMN NewDate DATE;

-- Step 2: Convert the existing text-based date format to 'YYYY-MM-DD'
UPDATE renewable_energy
SET NewDate = STR_TO_DATE(Date, '%W, %M %e, %Y');

-- Step 3: Remove the old 'Date' column and rename 'NewDate' to 'Date'
ALTER TABLE renewable_energy
DROP COLUMN Date,
CHANGE COLUMN NewDate Date DATE;


-- Step 1: Add a new column 'NewTime' with TIME data type
ALTER TABLE renewable_energy
ADD COLUMN NewTime TIME;

-- Step 2: Convert existing 'Time' values from 12-hour format to 24-hour format
UPDATE renewable_energy
SET NewTime = STR_TO_DATE(Time, '%h:%i:%s %p');

-- Step 3: Remove the old 'Time' column and rename 'NewTime' to 'Time'
ALTER TABLE renewable_energy
DROP COLUMN Time,
CHANGE COLUMN NewTime Time TIME;

-- Step 4: Modify table structure and update column data types in one step
ALTER TABLE renewable_energy
ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY,
MODIFY COLUMN EnergySource VARCHAR(50),
MODIFY COLUMN EnergyGenerated_kWh FLOAT,
MODIFY COLUMN Temperature_C FLOAT,
MODIFY COLUMN WindSpeed_mps FLOAT,
MODIFY COLUMN WeatherCondition VARCHAR(50),
MODIFY COLUMN Date DATE,
MODIFY COLUMN Time TIME;

describe renewableenergy.renewable_energy;