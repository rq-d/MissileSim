import csv

def parse(filename):
  data = {
      "time": [],
      "thrust": []
  }

  # Replace with your actual file name
  with open(filename, mode='r', newline='') as csvfile:
      reader = csv.reader(csvfile)
      
      for row in reader:
          if len(row) != 2:
              continue  # Skip malformed rows
          
          time_val = row[0].strip()
          thrust_val = row[1].strip()
          
          data["time"].append(float(time_val))
          data["thrust"].append(float(thrust_val))
  return data

