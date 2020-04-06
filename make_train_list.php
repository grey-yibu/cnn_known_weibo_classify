<?php

    $scan_dir  =  $argv[1];

    if(!isset($argv[1])){
        $scan_dir = "source";
    }

    $train_test_num = 0.75;
    $it = 0;
    $all_pic_info  =  array();


    $all_type = array("meinv"=>0,"zufang"=>1,"fengjing"=>2,"jianshen"=>3,"jiaotong"=>4,"xinmeishi"=>5,"yiqing"=>6);

    $scan_dir = trim($scan_dir,"\/");
    $type_list=scandir(__DIR__."/$scan_dir");
    foreach ($type_list  as $j){
         if($j=="."||$j==".."){
            continue;
        }
        $type_fold = trim($j,"\/");
        $list=scandir(__DIR__."/$scan_dir/$type_fold");
        echo  __DIR__."/$scan_dir/$type_fold"."\n";
        
        foreach ($list  as $i){
            if($i=="."||$i==".."){
                continue;
            }
            // 判断是否时图片
            if(!isImage( __DIR__."/$scan_dir/$type_fold/".$i)){
                continue;
            }
            $tmp_type = trim($type_fold,"\/");
            $tmp_type = $all_type[$tmp_type];
            $tmp_log = __DIR__."/$scan_dir/$type_fold/".$i." ".$tmp_type."\n";
            $all_pic_info[] = $tmp_log;
            // write2log( $out_f_name ,$tmp_log);
            $it++;
        }
    }

    //  随机打乱
    shuffle($all_pic_info);
    $real_count = count($all_pic_info);
    $train = array_slice($all_pic_info,0,$real_count*$train_test_num);
    $test  = array_slice($all_pic_info,$real_count*$train_test_num,$real_count);

    foreach ($train as $key => $value) {
        write2log("train.txt",$value);
    }
    foreach ($test as $key => $value) {
        write2log("test.txt",$value);
    }

function  write2log( $rizhi_name , $str  ){
    $file =  __DIR__."/".$rizhi_name;
    $fp = fopen($file, "a");
    fwrite($fp, $str);
    fclose($fp);
}

function isImage($filename)
{
 $types = '.gif|.jpg|.jpeg|.png|.bmp'; //定义检查的图片类型
 if(file_exists($filename))
 {
  if (($info = @getimagesize($filename)))
   return 1;
   
  $ext = image_type_to_extension($info['2']);
  return stripos($types,$ext);
 }
 else
 {
  return false;
 }
}



?>

