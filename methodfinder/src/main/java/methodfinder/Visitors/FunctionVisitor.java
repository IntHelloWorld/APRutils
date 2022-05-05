package methodfinder.Visitors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import methodfinder.Common.MethodContent;

/**
 * Visit MethodDeclaration and code with @Rule annotation, collect data
 * relations and method information.
 * 
 * @author Yihao Qin
 */
public class FunctionVisitor extends VoidVisitorAdapter<Object> {
	private ArrayList<MethodContent> m_Methods = new ArrayList<>();
	public List<String> targetMethodNames;

	public FunctionVisitor(List<String> targetMethodNames) {
		this.targetMethodNames = targetMethodNames;
	}

	// Collect method bodys with given method names
	@Override
	public void visit(MethodDeclaration node, Object arg) {
		if (targetMethodNames.contains(node.getName())) {
			if (node.getBody() != null) {
				m_Methods.add(new MethodContent(node, node.getName(), getMethodLength(node.getBody().toString())));
			}
		}
		super.visit(node, arg);
	}

	private long getMethodLength(String code) {
		String cleanCode = code.replaceAll("\r\n", "\n").replaceAll("\t", " ");
		if (cleanCode.startsWith("{\n"))
			cleanCode = cleanCode.substring(3).trim();
		if (cleanCode.endsWith("\n}"))
			cleanCode = cleanCode.substring(0, cleanCode.length() - 2).trim();
		if (cleanCode.length() == 0) {
			return 0;
		}
		long codeLength = Arrays.asList(cleanCode.split("\n")).stream()
				.filter(line -> (line.trim() != "{" && line.trim() != "}" && line.trim() != ""))
				.filter(line -> !line.trim().startsWith("/") && !line.trim().startsWith("*")).count();
		return codeLength;
	}

	public ArrayList<MethodContent> getMethodContents() {
		return m_Methods;
	}
}
