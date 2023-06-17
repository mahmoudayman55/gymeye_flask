-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jun 17, 2023 at 03:18 AM
-- Server version: 10.4.24-MariaDB
-- PHP Version: 8.1.6

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `gymeye`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--
CREATE DATABASE IF NOT EXISTS gymeye;
USE gymeye;
CREATE TABLE `admin` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL,
  `email` varchar(50) NOT NULL,
  `password` varchar(50) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `article`
--
--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL,
  `email` varchar(50) NOT NULL,
  `password` varchar(50) NOT NULL,
  `gender` varchar(50) NOT NULL,
  `age` int(11) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `article` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `author_id` int(11) NOT NULL,
  `category_id` int(11) NOT NULL,
  `title` varchar(20) NOT NULL,
  `description` varchar(400) NOT NULL,
  `image_url` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `author_id` (`author_id`),
  KEY `category_id` (`category_id`),
  KEY `author_id_2` (`author_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `bmi`
--

CREATE TABLE `bmi` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `gender` varchar(50) NOT NULL,
  `age` int(11) NOT NULL,
  `weight` int(11) NOT NULL,
  `water` int(11) NOT NULL,
  `protein` int(11) NOT NULL,
  `fat` int(11) NOT NULL,
  `daily_activity_level` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `category`
--
--
-- Table structure for table `exercise`
--

CREATE TABLE `exercise` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL,
  `description` varchar(400) NOT NULL,
  `video_url` varchar(100) NOT NULL,
  `cover_url` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `category` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `author_id` int(11) NOT NULL,
  `title` varchar(50) NOT NULL,
  `image_url` varchar(50) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `author_id` (`author_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------
-- --------------------------------------------------------

--
-- Table structure for table `evaluation`
--

CREATE TABLE `evaluation` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `exercise_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `date` date NOT NULL,
  PRIMARY KEY (`id`),
  KEY `exercise_id` (`exercise_id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `evaluation_ibfk_1` FOREIGN KEY (`exercise_id`) REFERENCES `exercise` (`id`),
  CONSTRAINT `evaluation_ibfk_2` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- --------------------------------------------------------

--
-- Table structure for table `exercise_error`
--
--
-- Table structure for table `evaluation_result`
--
select * from users;
#drop database gymeye;
CREATE TABLE `evaluation_result` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `evaluation_id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL,
  `description` varchar(400) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `evaluation_id` (`evaluation_id`),
  CONSTRAINT `evaluation_result_ibfk_1` FOREIGN KEY (`evaluation_id`) REFERENCES `evaluation` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `exercise_error` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `exercise_id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL,
  `description` varchar(400) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `exercise_id` (`exercise_id`),
  CONSTRAINT `exercise_error_ibfk_1` FOREIGN KEY (`exercise_id`) REFERENCES `exercise` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `exercise_error_results`
--

CREATE TABLE `exercise_error_results` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `exercise_error_id` int(11) NOT NULL,
  `evaluation_result_id` int(11) NOT NULL,
  `value` double NOT NULL,
  `screenshot` varchar(400) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `exercise_error_id` (`exercise_error_id`),
  KEY `evaluation_result_id` (`evaluation_result_id`),
  CONSTRAINT `exercise_error_results_ibfk_1` FOREIGN KEY (`exercise_error_id`) REFERENCES `exercise_error` (`id`),
  CONSTRAINT `exercise_error_results_ibfk_2` FOREIGN KEY (`evaluation_result_id`) REFERENCES `evaluation_result` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;



-- --------------------------------------------------------

-- --------------------------------------------------------


-- --------------------------------------------------------

--
-- Table structure for table `workout_plan`
--

CREATE TABLE `workout_plan` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `title` varchar(50) NOT NULL,
  `description` varchar(400) NOT NULL,
  `start_date` date NOT NULL,
  `end_date` date NOT NULL,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `workout_plan_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `workout_plan_exercise`
--

CREATE TABLE `workout_plan_exercise` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `workout_plan_id` int(11) NOT NULL,
  `exercise_id` int(11) NOT NULL,
  `sets` int(11) NOT NULL,
  `reps` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `workout_plan_id` (`workout_plan_id`),
  KEY `exercise_id` (`exercise_id`),
  CONSTRAINT `workout_plan_exercise_ibfk_1` FOREIGN KEY (`workout_plan_id`) REFERENCES `workout_plan` (`id`),
  CONSTRAINT `workout_plan_exercise_ibfk_2` FOREIGN KEY (`exercise_id`) REFERENCES `exercise` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `workout_plan_evaluation`
--

CREATE TABLE `workout_plan_evaluation` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `workout_plan_id` int(11) NOT NULL,
  `evaluation_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `workout_plan_id` (`workout_plan_id`),
  KEY `evaluation_id` (`evaluation_id`),
  CONSTRAINT `workout_plan_evaluation_ibfk_1` FOREIGN KEY (`workout_plan_id`) REFERENCES `workout_plan` (`id`),
  CONSTRAINT `workout_plan_evaluation_ibfk_2` FOREIGN KEY (`evaluation_id`) REFERENCES `evaluation` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `workout_plan_exercise_evaluation`
--

CREATE TABLE `workout_plan_exercise_evaluation` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `workout_plan_exercise_id` int(11) NOT NULL,
  `evaluation_result_id` int(11) NOT NULL,
  `value` double NOT NULL,
  PRIMARY KEY (`id`),
  KEY `workout_plan_exercise_id` (`workout_plan_exercise_id`),
  KEY `evaluation_result_id` (`evaluation_result_id`),
  CONSTRAINT `workout_plan_exercise_evaluation_ibfk_1` FOREIGN KEY (`workout_plan_exercise_id`) REFERENCES `workout_plan_exercise` (`id`),
  CONSTRAINT `workout_plan_exercise_evaluation_ibfk_2` FOREIGN KEY (`evaluation_result_id`) REFERENCES `evaluation_result` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;