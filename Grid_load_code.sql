SELECT * FROM renewableenergy.grid_load
limit 10;

describe renewableenergy.grid_load;

-- Step 1: Add new columns with correct data types for Date and Time
ALTER TABLE grid_load
ADD COLUMN NewDate DATE,
ADD COLUMN NewTime TIME;

-- Step 2: Convert existing text-based Date and Time to proper formats
UPDATE grid_load
SET NewDate = STR_TO_DATE(Date, '%W, %M %e, %Y'),
    NewTime = STR_TO_DATE(Time, '%h:%i:%s %p');

-- Step 3: Remove old Date and Time columns and rename new ones
ALTER TABLE grid_load
DROP COLUMN Date,
DROP COLUMN Time,
CHANGE COLUMN NewDate Date DATE,
CHANGE COLUMN NewTime Time TIME;

-- Step 4: Modify table structure to match the new format
ALTER TABLE grid_load
ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY,
MODIFY COLUMN GridLoad_MW FLOAT,
MODIFY COLUMN BatteryStorage_MWh FLOAT,
CHANGE COLUMN `Grid Stability Index` GridStabilityIndex FLOAT;
