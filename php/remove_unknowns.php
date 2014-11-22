<?php

/**
 * Phase 3 of cleaning the data: remove needless unkowns.
 * Generated cleaned CSVs that have (if possible) no "unknown" data
 * Problem data sets are:
 * 1-user
 * 2-business
 * 3-tip
 */
set_time_limit(0);
require('debug.php');

function get_infile($filename)
{
  return 'data/csv/'  . $filename;
}

function get_outfile($filename)
{
  return 'data/cleaned-csv/' . $filename;
}

// Remove all "unknowns" from user dataset (replace w/ 0)
// since in truth, all these things actually are known.
function generate_user_csv($filename)
{
  $infile = get_infile($filename);
  $outfile = get_outfile($filename);

  // sanitize: name, city, type, 
  $handle = fopen($infile, "r");  
  $handle_out = fopen($outfile, "w"); 
  
  $attributes = fgetcsv($handle, 0, ",",'"');
  $rs = fputcsv($handle_out, $attributes);

  while(($line = fgetcsv($handle, 0, ",",'"')) !== false) 
  {
    foreach ($line as $key => $val) 
    {
      // Don't allow any empty values
      if($line[$key] == "")
      {
        $line[$key] = "0";
      }
    }

    $rs = fputcsv($handle_out, $line);
  }

  fclose($handle);
  fclose($handle_out);
}

function generate_business_csv($filename)
{
  $infile = get_infile($filename);
  $outfile = get_outfile($filename);

  // sanitize: name, city, type, 
  $handle = fopen($infile, "r");  
  $handle_out = fopen($outfile, "w"); 
  
  $attributes = fgetcsv($handle, 0, ",",'"');
  $rs = fputcsv($handle_out, $attributes);
  // echo Debug::vars($attributes); exit;

  while(($line = fgetcsv($handle, 0, ",",'"')) !== false) 
  {
    foreach ($line as $key => $val) 
    {
      // This value seems to just imply "unknown"
      if($line[$key] == "{}")
      {
        $line[$key] = ""; 
      }

      // Assume attributes are UNKNOWN if not specified.
      // Assume categories & neighborhoods can ONLY be true or false; 
      if(stripos($attributes[$key], "categories") !== false OR stripos($attributes[$key], "neighborhoods") !== false)
      {
        if($line[$key] == "" OR $line[$key] == "{}")
        {
          // There's no half-way.
          // you're in a category, or you're not.
          // you're in a neighborhood, or you're not.
          $line[$key] = "F";
        }
      }

      if($line[$key] == "True")
      {
        $line[$key] = "T";
      }

      if($line[$key] == "False")
      {
        $line[$key] = "F";
      }
    }

    $rs = fputcsv($handle_out, $line);
  }

  fclose($handle);
  fclose($handle_out);
}

function generate_tip_csv($filename)
{
  $infile = get_infile($filename);
  $outfile = get_outfile($filename);

  // sanitize: name, city, type, 
  $handle = fopen($infile, "r");  
  $handle_out = fopen($outfile, "w"); 
  
  $attributes = fgetcsv($handle, 0, ",",'"');
  $rs = fputcsv($handle_out, $attributes);
  echo Debug::vars($attributes); exit;

  while(($line = fgetcsv($handle, 0, ",",'"')) !== false) 
  {
    foreach ($line as $key => $val) 
    {
      // Assume attributes are UNKNOWN if not specified.
      // Assume categories & neighborhoods can ONLY be true or false; 
      if(stripos($attributes[$key], "categories") !== false OR stripos($attributes[$key], "neighborhoods") !== false)
      {
        if($line[$key] == "" OR $line[$key] == "{}")
        {
          // There's no half-way.
          // you're in a category, or you're not.
          // you're in a neighborhood, or you're not.
          $line[$key] = "F";
        }
      }

      if($line[$key] == "True")
      {
        $line[$key] = "T";
      }

      if($line[$key] == "False")
      {
        $line[$key] = "F";
      }
    }

    $rs = fputcsv($handle_out, $line);
  }

  fclose($handle);
  fclose($handle_out);
}

function generate_review_csv($filename)
{
  // Nothing to do
}

function generate_checkin_csv($filename)
{
  // Nothing to do  
}


echo "<pre>starting task\n</pre>";
// generate_user_csv('yelp_academic_dataset_user.csv');
// generate_business_csv('yelp_academic_dataset_business.csv');
// generate_checkin_csv('yelp_academic_dataset_checkin.csv'); 
// generate_review_csv('yelp_academic_dataset_review.csv'); 
// generate_tip_csv('yelp_academic_dataset_tip.csv'); 
echo "<pre>All done! Generated CSV.\n</pre>";
