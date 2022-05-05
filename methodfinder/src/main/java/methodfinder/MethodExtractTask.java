package methodfinder;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import com.github.javaparser.ParseException;
import com.github.javaparser.ast.CompilationUnit;

import methodfinder.Common.CommandLineValues;
import methodfinder.Common.Common;
import methodfinder.Common.Line;
import methodfinder.Common.MethodContent;
import methodfinder.Common.ProgressBar;
import methodfinder.Visitors.FunctionVisitor;

public class MethodExtractTask implements Callable<Void> {
	CommandLineValues m_CommandLineValues;
	Path filePath;
	Path classPath;
	String projectName;
	String label;
	List<String> methodNames;
	ProgressBar progressBar = null;

	public MethodExtractTask(CommandLineValues commandLineValues, Line currentLine, ProgressBar progressBar) {
		m_CommandLineValues = commandLineValues;
		this.classPath = currentLine.classPath;
		this.projectName = currentLine.projectName;
		this.filePath = m_CommandLineValues.projectDir.resolve(classPath);
		this.label = currentLine.label;
		this.methodNames = currentLine.targetMethodNames;
		this.progressBar = progressBar;
	}

	@Override
	public Void call() throws Exception {
		processFile();
		return null;
	}

	public void processFile() throws InterruptedException {
		ArrayList<MethodContent> methods;
		try {
			methods = extractSingleFile();
			DatasetGenerator.generateDataset(methods, m_CommandLineValues, classPath, projectName, label, progressBar);
		} catch (ParseException | IOException e) {
			e.printStackTrace();
			return;
		}
		if (methods == null) {
			return;
		}
	}

	public void SaveFeaturesToFile(Path filename, String content) {
		try {
			FileWriter fw = new FileWriter(filename.toString(), false);
			fw.write(content);
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public ArrayList<MethodContent> extractSingleFile() throws ParseException, IOException {
		String code = null;
		try {
			code = new String(Files.readAllBytes(this.filePath));
		} catch (IOException e) {
			e.printStackTrace();
			code = Common.EmptyString;
		}
		CompilationUnit compilationUnit = Common.parseFileWithRetries(code);
		FunctionVisitor functionVisitor = new FunctionVisitor(this.methodNames);
		functionVisitor.visit(compilationUnit, null);
		return functionVisitor.getMethodContents();

	}
}
