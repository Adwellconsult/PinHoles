[app]

# (str) Title of your application
title = Bullet Impact Detector

# (str) Package name
package.name = bulletdetector

# (str) Package domain (needed for android/ios packaging)
package.domain = com.pinholes.bulletdetector

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,json,md

# (list) Source files to exclude (let empty to not exclude anything)
source.exclude_exts = spec

# (list) List of directory to exclude (let empty to not exclude anything)
source.exclude_dirs = tests, bin, .buildozer, .git, __pycache__

# (list) List of exclusions using pattern matching
source.exclude_patterns = license,*.txt,*.md,requirements.txt

# (str) Application versioning (method 1)
version = 1.0

# (str) Application versioning (method 2)
# version.regex = __version__ = ['"](.*?)['"] 
# version.filename = %(source.dir)s/main.py

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3,kivy==2.3.1,opencv-python==4.8.1.78,numpy,pillow,plyer,pyjnius,android

# (str) Presplash of the application
#presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon of the application
#icon.filename = %(source.dir)s/data/icon.png

# (str) Supported orientation (portrait, sensorPortrait, landscape, sensorLandscape)
orientation = landscape

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (string) Presplash of the application
#presplash.filename = %(source.dir)s/data/presplash.png

# (string) Icon of the application
#icon.filename = %(source.dir)s/data/icon.png

# (list) Permissions
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,INTERNET,ACCESS_NETWORK_STATE

# (list) Android application meta-data to set (key=value format)
#android.meta_data =

# (list) Android library project to add (will be added in the
# project.properties automatically.)
#android.library_references = @jar/my-lib-1.jar,@jar/my-lib-2.jar

# (int) Target Android API, should be as high as possible.
android.api = 33

# (int) Minimum API your APK / AAB will support.
android.minapi = 21

# (str) Android NDK version to use
android.ndk = 25b

# (str) Android SDK version to use
android.sdk = 33

# (str) python-for-android fork to use
# p4a.fork = kivy

# (str) python-for-android branch to use
# p4a.branch = main

# (str) python-for-android specific commit to use
# p4a.commit = HEAD

# (str) python-for-android git clone directory (if empty, it will be automatically cloned from github)
#p4a.source_dir =

# (str) The main.py to use
# If not specified, buildozer will look for main.py in source.dir
# This allows you to specify a different main file
source.main = bullet_detector_android.py

# (str) Bootstrap to use for android builds
# p4a.bootstrap = sdl2

# (str) Android entry point, default is ok for Kivy-based app
#android.entrypoint = org.kivy.android.PythonActivity

# (str) Full name of the Java class that implements Python Service
# service.main_class = org.kivy.android.PythonService

# (str) Android app theme, default is ok for Kivy-based app
# android.theme = @android:style/Theme.NoTitleBar

# (int) port number to specify an explicit --port= p4a argument (eg for bootstrap flask)
#p4a.port =

# Control passing the --private-data-dir argument to p4a
#p4a.private_data_dir = True

# Pass --use-setup-py to p4a
#p4a.setup_py = false

# Pass --ignore-setup-py to p4a
#p4a.ignore_setup_py = false

# (str) Android logcat filters to use
android.logcat_filters = *:S python:D

# (bool) Copy library instead of making a libpymodules.so
#android.copy_libs = 1

# (list) The Android archs to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.archs = arm64-v8a, armeabi-v7a

# (bool) enables Android auto backup feature (Android API >=23)
android.allow_backup = True

# (str) The format used to package the app for debug mode (apk or aab).
android.debug_artifact = apk

# (str) The format used to package the app for release mode (aab or apk).
android.release_artifact = apk

# (bool) Skip byte compile for .py files
android.no-byte-compile-python = False

# (str) Android gradle dependencies to add
#android.gradle_dependencies =

# (bool) Enable AndroidX support. Enable when 'android.gradle_dependencies'
# contains an 'androidx' package, or any package from Kotlin source.
#android.enable_androidx = False

# (str) Android add java compile options
#android.add_compile_options = "sourceCompatibility = 1.8", "targetCompatibility = 1.8"

# (list) Gradle repositories to add
#android.gradle_repositories = google(), mavenCentral()

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1

# (str) Path to build artifact storage, absolute or relative to spec file
# build_dir = ./.buildozer

# (str) Path to build output (i.e. .apk, .aab, .ipa) storage
# bin_dir = ./bin

#    -----------------------------------------------------------------------------
#    List as sections
#
#    You can define all the "list" as [section:key].
#    Each line will be considered as a option to the list.
#    Let's take [app] / source.exclude_patterns.
#    Instead of doing:
#
#[app]
#source.exclude_patterns = license,data/audio/*.wav,data/images/original/*
#
#    This can be translated into:
#
#[app:source.exclude_patterns]
#license
#data/audio/*.wav
#data/images/original/*
#

#    -----------------------------------------------------------------------------
#    Profiles
#
#    You can extend section / key with a profile
#    For example, you want to deploy a demo version of your application without
#    HD content. You could first change the title to add "(demo)" in the name
#    and extend the excluded directories to remove the HD content.
#
#[app@demo]
#title = My Application (demo)
#
#[app:source.exclude_patterns@demo]
#images/hd/*
#
#    Then, invoke the command line with the "demo" profile:
#
#buildozer --profile demo android debug